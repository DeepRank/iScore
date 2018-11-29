from iScore.graphrank.graph import Graph
from iScore.graphrank.kernel import Kernel
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
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

def kmat_cpu(G1,G2,lamb=1, walk=4, method='vect'):

    #kernel routine
    ker = Kernel()
    t0 = time()
    ker.compute_kron_mat(G1,G2)
    return ker.Wx

def kmat_cuda(G1,G2,kernel_name, lamb=1, walk=4, ):

    #kernel routine
    ker = Kernel()
    ker.compile_kernel()
    t0 = time()
    ker.compute_kron_mat_cuda(G1,G2,kernel_name=kernel_name)
    return ker.Wx

def kmat_cublas(G1,G2,lamb=1, walk=4):

    #kernel routine
    ker = Kernel()
    ker.compile_kernel()
    t0 = time()
    ker.compute_kron_mat_cublas(G1,G2)
    return ker.Wx



size = 32
num_nodes  = [size,size]
edge_proba = 6./size

G1 = generate_graph('g1',num_nodes, edge_proba)
G2 = generate_graph('g2',num_nodes, edge_proba)
print('G1 : Number of Nodes : %d Number of Edges : %d' %(G1.num_nodes,G1.num_edges))
print('G2 : Number of Nodes : %d Number of Edges : %d' %(G2.num_nodes,G2.num_edges))


wcpu = kmat_cpu(G1,G2,lamb=1, walk=4, method='vect')
wgpu= kmat_cuda(G1,G2,kernel_name='create_kron_mat')
wgpu_shared = kmat_cuda(G1,G2,kernel_name='create_kron_mat_shared')



assert(np.allclose(wcpu.data,wgpu.data))
assert(np.allclose(wcpu.data,wgpu_shared.data))





