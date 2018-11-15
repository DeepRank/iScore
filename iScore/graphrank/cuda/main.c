#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <sys/time.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/betweenness_centrality.hpp>
#include <boost/graph/random.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <random>

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



void  create_graph(int *edges_index, float *nodes_data)
{


}

int main(int argc, char **argv)
{

    printf("Starting ...\n");

    // set up the device
    int dev = 0;
    struct cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using device %d : %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

}
