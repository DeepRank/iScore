from iScore.graphrank.graph import Graph
from iScore.graphrank.kernel import Kernel
import networkx as nx
from networkx.algorithms import bipartite

try:
    import pycuda.autoinit
    from pycuda import driver, compiler, gpuarray, tools
    from pycuda.reduction import ReductionKernel

except ModuleNotFoundError:
    print('Warning : pycuda not found')

import numpy as np
from time import time

def generate_graph(name,num_nodes,edge_proba):
    _tmp = bipartite.random_graph(num_nodes[0],num_nodes[1],edge_proba)
    edges = np.array([list(e) for e in _tmp.edges])
    pssm = np.random.randint(-10,10,size=(np.sum(num_nodes),20))
    info = np.random.rand(np.sum(num_nodes))
    return Graph(name=name,nodes_pssm=pssm,nodes_info=info,edges_index=edges)

def test_cpu(G1,G2,lamb=1, walk=4, method='vect'):

    #kernel routine
    ker = Kernel()
    t0 = time()

    ker.compute_kron_mat(G1,G2)
    ker.compute_px(G1,G2)
    ker.compute_W0(G1,G2)

    # compute the graphs
    ker.compute_K(lamb=lamb,walk=walk)
    print("Total time : %f\n" %(time()-t0))

def test_gpu(G1,G2,lamb=1, walk=4):

    #kernel routine
    ker = Kernel()
    t0 = time()

    # compile the kernel
    ker.compile_kernel()

    # prebook the weight and index matrix
    # required for the calculation of self.Wx
    t0 = time()
    n1 = G1.num_edges
    n2 = G2.num_edges
    n_edges_prod = 2*n1*n2
    ker.weight_product = gpuarray.zeros(n_edges_prod, np.float32)
    ker.index_product = gpuarray.zeros((n_edges_prod,2), np.int32)
    print('GPU - Mem  : %f' %(time()-t0))

    ker.compute_kron_mat_cuda(G1,G2)
    ker.compute_px_cuda(G1,G2)
    ker.compute_W0_cuda(G1,G2)

    # compute the graphs
    ker.compute_K(lamb=lamb,walk=walk)
    print("Total time : %f\n" %(time()-t0))

if __name__ == "__main__":


    # generate graphs
    num_nodes  = [100,100]
    edge_proba = 0.5
    G1 = generate_graph('g1',num_nodes, edge_proba)
    G2 = generate_graph('g2',num_nodes, edge_proba)

    #test_cpu(G1,G2)
    test_gpu(G1,G2)






