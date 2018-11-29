#include <cuda.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#include <boost/random.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/random.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/undirected_graph.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <random>


#include "kernel.hpp"




#define LEN 20
#define LEN2 2*LEN

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        printf("Error : %s : %d ", __FILE__, __LINE__);\
        printf("code %d, reason %s", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}\

struct VertexProperty
{
    float pssm[LEN];
};

using namespace boost;
using namespace boost::numeric::ublas;
using namespace std;

typedef boost::mt19937 RNGType;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperty> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor vertex_t;
typedef boost::graph_traits <Graph>::edge_iterator edgeIt;
typedef boost::graph_traits<Graph>::vertex_iterator vertexIt;
typedef map<vertex_t,size_t> IndexMap;

typedef boost::numeric::ublas::matrix<float> matf;
typedef boost::numeric::ublas::matrix<int> mati;

Graph create_graph(int num_node, int num_edge, RNGType rng)
{
    // boost random graph
    Graph g(num_node);
    boost::generate_random_graph(g, num_node, num_edge, rng, true, false);
    
    // add pssm on nodes
    vertexIt vitr, vend;
    boost::tie(vitr,vend) = boost::vertices(g);
    for (;vitr!=vend;++vitr){
        for (int iL =0 ; iL<LEN; iL++)
            g[*vitr].pssm[iL] = rand()%20 - 10;
    }
    return g;
}

void create_node_data(std::vector<float> &node_info, int num_node)
{
    for (int i=0; i < num_node; i++)
        node_info[i] = rand() % 1; 
}

void extract_graph_data(std::vector<int> &edge_index, std::vector<float> &pssm, Graph g)
{

    edgeIt eitr, eend;
    boost::tie(eitr,eend) = boost::edges(g);
    int iedge = 0;
    for(;eitr!=eend;++eitr)
    {
        edge_index[iedge*2]   = source(*eitr,g);
        edge_index[iedge*2+1] = target(*eitr,g);

        for(int iL=0;iL<LEN;iL++)
        {
            pssm[iedge*LEN2+iL]     = g[source(*eitr,g)].pssm[iL];
            pssm[iedge*LEN2+iL+LEN] = g[target(*eitr,g)].pssm[iL];
        }
        iedge++;
    }
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return( (double)tp.tv_sec + (double)tp.tv_usec*1.e-6 );
}

int main(int argc, char **argv)
{
    bool __print_device__ = false;
    int dimx = 8, dimy = 8;
    if (argc > 2)
    {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    printf("Starting ...\n");
    RNGType rng;

    // set up the device
    int dev = 0;
    struct cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using device %d : %s\n", dev, deviceProp.name);
    if (__print_device__)
    {
        printf("Number of multiprocessor %d\n",deviceProp.multiProcessorCount);
        printf("Total amount of constant memory: %4.2f KB\n",deviceProp.totalConstMem/1024.0);
        printf("Total amount of shared memory per block: %4.2f KB\n",deviceProp.sharedMemPerBlock/1024.0);
        printf("Wrap size: %d\n", deviceProp.warpSize);
        printf("Maximum number of thread per block: %d\n", deviceProp.maxThreadsPerBlock);
    }
    CHECK(cudaSetDevice(dev));

    // generate random graphs
    int num_node_1 = 100;
    int num_edge_1 = 1000;
    Graph g1 = create_graph(num_node_1,num_edge_1,rng);   
    //boost::print_graph(g1);

    // generatee random graphs
    int num_node_2 = 200;
    int num_edge_2 = 2000;
    Graph g2 = create_graph(num_node_2,num_edge_2,rng);
    //boost::print_graph(g2);
    
    // extract graph data
    std::vector<int> edge_index_1(2*num_edge_1);
    std::vector<float> edge_pssm_1(LEN2*num_edge_1);
    extract_graph_data(edge_index_1, edge_pssm_1, g1);

    // extract graph data
    std::vector<int> edge_index_2(2*num_edge_2);
    std::vector<float> edge_pssm_2(LEN2*num_edge_2);
    extract_graph_data(edge_index_2, edge_pssm_2, g2);

    //allocate device global data : graph 1
    // edges
    int *d_edge_index_1;
    size_t nBytes_index1 = 2*num_edge_1*sizeof(int);
    cudaMalloc((void**)&d_edge_index_1,nBytes_index1);
    cudaMemcpy(d_edge_index_1, edge_index_1.data(), nBytes_index1,cudaMemcpyHostToDevice);

    // pssm
    float *d_edge_pssm_1;
    size_t nBytes_pssm1 = LEN2 * num_edge_1 * sizeof(float);
    cudaMalloc((void**)&d_edge_pssm_1,nBytes_pssm1);
    cudaMemcpy(d_edge_pssm_1, edge_pssm_1.data(),nBytes_pssm1,cudaMemcpyHostToDevice);

    //allocate device global data : graph 2
    // pssm
    int *d_edge_index_2;
    size_t nBytes_index2 = 2*num_edge_2*sizeof(int);
    cudaMalloc((void**)&d_edge_index_2,nBytes_index2);
    cudaMemcpy(d_edge_index_2, edge_index_2.data(),nBytes_index2,cudaMemcpyHostToDevice);

    // pssm
    float *d_edge_pssm_2;
    size_t nBytes_pssm2 = LEN2*num_edge_2*sizeof(float);
    cudaMalloc((void**)&d_edge_pssm_2,nBytes_pssm2);
    cudaMemcpy(d_edge_pssm_2, edge_pssm_2.data(), nBytes_pssm2, cudaMemcpyHostToDevice);

    // allocate global device memory : output
    int *d_edge_index_prod;
    size_t nBytes_index_prod = 4*num_edge_2*num_edge_1*sizeof(int);
    cudaMalloc((void**)&d_edge_index_prod,nBytes_index_prod);

    float *d_edge_weight_prod;
    size_t nBytes_weight_prod = 2*num_edge_2*num_edge_1*sizeof(float);
    cudaMalloc((void**)&d_edge_weight_prod,nBytes_weight_prod);

    //set up the grid
    dim3 block(dimx,dimy);
    dim3 grid( (num_edge_1+block.x-1)/block.x, (num_edge_2+block.y-1)/block.y );  

    // comoute the kronecker matrix  
    double iStart = cpuSecond();
    create_kron_mat_shared<<< grid, block >>>(d_edge_index_1,d_edge_index_2, \
                                       d_edge_pssm_1, d_edge_pssm_2,  \
                                       d_edge_index_prod, d_edge_weight_prod,
                                       num_edge_1, num_edge_2, num_node_2 );
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;
    printf("Kron_Mat <<< (%d,%d), (%d,%d) >>> : %f sec.\n", \
            grid.x, grid.y, block.x, block.y, iElaps);



    // create node data for testing px
    std::vector<float> node_data_1(num_node_1);
    std::vector<float> node_data_2(num_node_2);
    std::vector<float> pvect(num_node_1,num_node_2);
    create_node_data(node_data_1,num_node_1);
    create_node_data(node_data_2,num_node_2);

    // node data on the PGU
    float *d_node_data_1;
    size_t nBytes_node1 = num_node_1 * sizeof(float);
    cudaMalloc((void**)&d_node_data_1,nBytes_node1);
    cudaMemcpy(d_node_data_1, node_data_1.data(), nBytes_node1, cudaMemcpyHostToDevice);

    // node data on the PGU
    float *d_node_data_2;
    size_t nBytes_node2 = num_node_2 * sizeof(float);
    cudaMalloc((void**)&d_node_data_2,nBytes_node2);
    cudaMemcpy(d_node_data_2, node_data_2.data(), nBytes_node2, cudaMemcpyHostToDevice);

    float *d_pvect;
    size_t nBytes_pvect = num_node_1*num_node_2;
    cudaMalloc((void**)&d_pvect,nBytes_pvect);


    iStart = cpuSecond();
    create_p_vect<<< grid, block >>>(d_node_data_1, d_node_data_2, d_pvect, num_node_1, num_node_2);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("create_pvect <<< (%d,%d), (%d,%d) >>> : %f sec.\n", \
            grid.x, grid.y, block.x, block.y, iElaps);

    cudaFree(d_edge_index_1);
    cudaFree(d_edge_pssm_1);
    cudaFree(d_edge_index_2);
    cudaFree(d_edge_pssm_2);
    cudaFree(d_edge_index_prod);
    cudaFree(d_edge_weight_prod);

    cudaFree(d_node_data_1);
    cudaFree(d_node_data_2);
    cudaFree(d_pvect);

    cudaDeviceReset();
    return 0;

}
