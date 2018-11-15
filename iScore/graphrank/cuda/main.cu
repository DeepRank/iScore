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



__global__ void create_kron_mat( int *edges_index_1, int *edges_index_2, 
                                 float *edges_pssm_1, float *edges_pssm_2, 
                                 int *edges_index_product, float *edges_weight_product,
                                 int n_edges_1, int n_edges_2,
                                 int n_nodes_2);

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

    int dimx = 8;
    int dimy = 8;
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
    CHECK(cudaSetDevice(dev));

    // generate random graphs
    int num_node_1 = 100;
    int num_edge_1 = 1000;
    Graph g1 = create_graph(num_node_1,num_edge_1,rng);   
    //boost::print_graph(g1);

    // generatee random graphs
    int num_node_2 = 100;
    int num_edge_2 = 1000;
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
    double iStart = cpuSecond();
    create_kron_mat<<< grid, block >>>(d_edge_index_1,d_edge_index_2, \
                                       d_edge_pssm_1, d_edge_pssm_2,  \
                                       d_edge_index_prod, d_edge_weight_prod,
                                       num_edge_1, num_edge_2, num_node_2 );
    cudaDeviceSynchronize();
    double iElaps = cpuSecond() - iStart;
    printf("Kron_Mat <<< (%d,%d), (%d,%d) >>> : %f sec.\n", \
            grid.x, grid.y, block.x, block.y, iElaps);

    cudaFree(d_edge_index_1);
    cudaFree(d_edge_pssm_1);
    cudaFree(d_edge_index_2);
    cudaFree(d_edge_pssm_2);
    cudaFree(d_edge_index_prod);
    cudaFree(d_edge_weight_prod);

}
