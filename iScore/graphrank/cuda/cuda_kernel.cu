#include <math.h>
#include <cuda_runtime.h>
// the rbf kernel function
__host__ __device__ float rbf_kernel(int tx, int ty, float *a, float *b, int len, int invert)
{
	float sigma = 10.;
	float beta = 0.5/sigma/sigma;
	float d = 0;
	float k = 0;
	float x,y;
	if(invert == 0)
	{
		for(int i=0;i<len;i++)
		{
			x = a[tx*len+i];
			y = b[ty*len+i];
			d += (x-y)*(x-y);
		}
	}

	else if (invert == 1)
	{
		int half = len/2;
		for(int i=0;i<half;i++)
		{
			x  = a[tx*len+i+half];
			y = b[ty*len+i];
			d += (x-y)*(x-y);

			x = a[tx*len+i];
			y = b[ty*len+i+half];
			d += (x-y)*(x-y);
			// d += (a[tx*len+i+half]-b[ty*len+i])*(a[tx*len+i+half]-b[ty*len+i]);
			// d += (a[tx*len+i]-b[ty*len+i+half])*(a[tx*len+i]-b[ty*len+i+half]);
		}
	}

	k = exp(-beta*d);
	return k;

}


__global__ void create_kron_mat( int *edges_index_1, int *edges_index_2,
                                 float *edges_pssm_1, float *edges_pssm_2,
                                 int *edges_index_product, float *edges_weight_product,
                                 int n_edges_1, int n_edges_2,
                                 int n_nodes_2)
{

	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	int ind=0, len = 40;
	float w;
	int invert;

	if ( (tx < n_edges_1) && (ty < n_edges_2) ){

		////////////////////////////////////
		// first pass
		// i-j VS a-b
		////////////////////////////////////

		// get the index of the element
		ind = tx * n_edges_2 + ty;

		// get its weight
		invert=0;
		w = rbf_kernel(tx,ty,edges_pssm_1,edges_pssm_2,len,invert);

		// store it
		edges_weight_product[ind]       = w;
		edges_index_product[2*ind]      = edges_index_1[2*tx] * n_nodes_2 + edges_index_2[2*ty] ;
		edges_index_product[2*ind + 1]  = edges_index_1[2*tx+1] * n_nodes_2   + edges_index_2[2*ty+1] ;

		////////////////////////////////////
		// second pass
		// j-i VS a-b
		////////////////////////////////////

		// get the index element
		ind = ind + n_edges_1 * n_edges_2;

		// get the weight
		invert=1;
		w = rbf_kernel(tx,ty,edges_pssm_1,edges_pssm_2,len,invert);

		// store it
		edges_weight_product[ind]       = w;
		edges_index_product[2*ind]      = edges_index_1[2*tx+1]   * n_nodes_2   + edges_index_2[2*ty];
		edges_index_product[2*ind + 1]  = edges_index_1[2*tx]     * n_nodes_2   + edges_index_2[2*ty+1];

	}
}



// the rbf kernel function
__host__ __device__ float rbf_kernel_shared(int tx, int ty, float *tile, int len, int invert, const int halfsdimx)
{
	float sigma = 10.;
	float beta = 0.5/sigma/sigma;
	float d = 0;
	float k = 0;
	float x=0,y=0; 
	int i;

	if(invert == 0)
	{
		for(i=0;i<len;++i)
		{
			// ind = tx*len+i;
			// if (ind <640)
			// 	x = tile[ind];
			// else{
			// 	x = 0;
			// 	printf("Wrong X index for thread %d %d\n", tx,ty);
			// }

			// ind = (ty+halfsdimx)*len + i;
			// if (ind <640)
			// 	y = tile[ind];
			// else{
			// 	y = 0;
			// 	printf("Wrong Y index for thread %d %d i=%d : ind = %d\n", tx,ty,i, ind);
			// }
			x = tile[tx*len+i];
			y = tile[ (ty+halfsdimx)*len + i];

			d += (x-y)*(x-y);
		}
	}

	else if (invert == 1)
	{
		int half = len/2;
		for(int i=0;i<half;i++)
		{
			x = tile[tx*len+i+half];
			y = tile[(ty+halfsdimx)*len+i];
			d += (x-y)*(x-y);

			x = tile[tx*len+i];
			y = tile[(ty+halfsdimx)*len + i + half];
			d += (x-y)*(x-y);
		}
	}

	k = exp(-beta*d);
	return k;

}

__global__ void create_kron_mat_shared( int *edges_index_1, int *edges_index_2, 
                                 float *edges_pssm_1, float *edges_pssm_2, 
                                 int *edges_index_product, float *edges_weight_product,
                                 int n_edges_1, int n_edges_2,
                                 int n_nodes_2)
{
	
	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	int ind=0, len = 40;
	float w=0;
	int invert;


	const int SDIMX = 16;
	const int HALF_SDIMX = 8;
	const int SDIMY = 40;


	__shared__ float tile[SDIMX*SDIMY];

	if ( (tx < n_edges_1) && (ty < n_edges_2) ){

		////////////////////////////////////
		// first pass
		// i-j VS a-b
		////////////////////////////////////

		// get the index of the element
		ind = tx * n_edges_2 + ty;

		// populate shared memory
		for (int i=0; i<len;i++)
		{
			tile[threadIdx.x*len + i]            =  edges_pssm_1[tx*len+i];
			tile[(threadIdx.y+HALF_SDIMX)*len+i] =  edges_pssm_2[ty*len+i];
		}
		__syncthreads();

		// get its weight
		invert=0;
		w = rbf_kernel_shared(threadIdx.x,threadIdx.y,tile,len,invert,HALF_SDIMX);

		// store it
		edges_weight_product[ind]       = w;
		edges_index_product[2*ind]      = edges_index_1[2*tx] * n_nodes_2 + edges_index_2[2*ty] ;
		edges_index_product[2*ind + 1]  = edges_index_1[2*tx+1] * n_nodes_2   + edges_index_2[2*ty+1] ;

		////////////////////////////////////
		// second pass
		// j-i VS a-b
		////////////////////////////////////

		// get the index element
		ind = ind + n_edges_1 * n_edges_2;

		// get the weight
		invert=1;
		w = rbf_kernel_shared(threadIdx.x,threadIdx.y,tile,len,invert,HALF_SDIMX);

		// store it
		edges_weight_product[ind]       = w;
		edges_index_product[2*ind]      = edges_index_1[2*tx+1]   * n_nodes_2   + edges_index_2[2*ty];
		edges_index_product[2*ind + 1]  = edges_index_1[2*tx]     * n_nodes_2   + edges_index_2[2*ty+1];

	}
}



__global__ void create_nodesim_mat(float *nodes_pssm_1, float *nodes_pssm_2, float *W0, int n_nodes_1, int n_nodes_2)
{

	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	int len,ind;

	if ( (tx<n_nodes_1) && (ty<n_nodes_2))
	{

		len = 20;
		ind = tx * n_nodes_2 + ty;
		int invert = 0;
		float sim;

		sim = rbf_kernel(tx,ty,nodes_pssm_1,nodes_pssm_2,len,invert);
		W0[ind] = sim;
	}
}

__global__ void create_p_vect(float *node_info1, float* node_info2, float *p, int n_nodes_1, int n_nodes_2)
{

	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;
	float cutoff = 0.5;

	if ( (tx < n_nodes_1) && (ty < n_nodes_2) )
	{
		int ind = tx * n_nodes_2 + ty;
		if ( (node_info1[tx] < cutoff) && (node_info2[ty] < cutoff))
			p[ind] = 0;
		else
			p[ind] = node_info1[tx] * node_info2[ty];
	}
}
