#ifndef __KERNEL_CU__
#define __KERNEL_CU__
__global__ void create_kron_mat( int *edges_index_1, int *edges_index_2, 
                                 float *edges_pssm_1, float *edges_pssm_2, 
                                 int *edges_index_product, float *edges_weight_product,
                                 int n_edges_1, int n_edges_2,
                                 int n_nodes_2);

__global__ void create_p_vect(float *node_info1, float* node_info2, float *p, int n_nodes_1, int n_nodes_2);

__global__ void create_kron_mat_shared( int *edges_index_1, int *edges_index_2, 
                                 float *edges_pssm_1, float *edges_pssm_2, 
                                 int *edges_index_product, float *edges_weight_product,
                                 int n_edges_1, int n_edges_2,
                                 int n_nodes_2);

#endif