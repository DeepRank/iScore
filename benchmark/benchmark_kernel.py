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

def time_kmat_cpu(G1,G2,lamb=1, walk=4, method='vect'):

    #kernel routine
    ker = Kernel()
    t0 = time()
    ker.compute_kron_mat(G1,G2)
    return time() - t0

def time_kmat_cuda(G1,G2,lamb=1, walk=4):

    #kernel routine
    ker = Kernel()
    ker.compile_kernel()
    t0 = time()
    ker.compute_kron_mat_cuda(G1,G2)
    return time() - t0

def time_kmat_cublas(G1,G2,lamb=1, walk=4):

    #kernel routine
    ker = Kernel()
    ker.compile_kernel()
    t0 = time()
    ker.compute_kron_mat_cublas(G1,G2)
    return time() - t0


if __name__ == "__main__":

    num_rep = 1
    time_cpu_iter, time_cpu_comb,  time_cpu_vect = [], [], []
    time_cuda, time_cublas = [], []

    mean_time_cpu_iter, mean_time_cpu_comb, mean_time_cpu_vect = [], [], []
    mean_time_cuda, mean_time_cublas = [], []
    
    sizes = [2**2,2**3,2**4,2**5,2**6,2**7,2**8,2**9,2**10,2**11]

    for iS,s in enumerate(sizes):

        # generate graphs
        num_nodes  = [s,s]
        edge_proba = 6./s

        time_cpu_iter.append([])
        time_cpu_comb.append([])
        time_cpu_vect.append([])
        time_cuda.append([])
        time_cublas.append([])
      

        for _ in range(num_rep):

            G1 = generate_graph('g1',num_nodes, edge_proba)
            G2 = generate_graph('g2',num_nodes, edge_proba)
            print('G1 : Number of Nodes : %d Number of Edges : %d' %(G1.num_nodes,G1.num_edges))
            print('G2 : Number of Nodes : %d Number of Edges : %d' %(G2.num_nodes,G2.num_edges))

            time_cpu_comb[iS].append(time_kmat_cpu(G1,G2,method='combvec'))
            time_cpu_iter[iS].append(time_kmat_cpu(G1,G2,method='iter'))
            time_cpu_vect[iS].append(time_kmat_cpu(G1,G2,method='vect'))

            time_cuda[iS].append(time_kmat_cuda(G1,G2))
            time_cublas[iS].append(time_kmat_cublas(G1,G2))

        mean_time_cpu_comb.append(np.mean(time_cpu_comb[iS]))
        mean_time_cpu_iter.append(np.mean(time_cpu_iter[iS]))
        mean_time_cpu_vect.append(np.mean(time_cpu_vect[iS]))

        mean_time_cuda.append(np.mean(time_cuda[iS]))
        mean_time_cublas.append(np.mean(time_cublas[iS]))


    np.savetxt('sizes.dat',np.array(sizes))

    np.savetxt('mean_time_cpu_iter.dat',np.array(mean_time_cpu_iter))
    np.savetxt('mean_time_cpu_comb.dat',np.array(mean_time_cpu_comb))
    np.savetxt('mean_time_cpu_vect.dat',np.array(mean_time_cpu_vect))
    np.savetxt('mean_time_cuda.dat',np.array(mean_time_cuda))
    np.savetxt('mean_time_cublas.dat',np.array(mean_time_cublas))

    sizes = np.array(sizes)
    plt.plot(2*sizes,mean_time_cpu_iter,'o-',label='CPU iter')
    plt.plot(2*sizes,mean_time_cpu_comb,'o-',label='CPU comb')
    plt.plot(2*sizes,mean_time_cpu_vect,'o-',label='CPU vect')
    plt.plot(2*sizes,mean_time_cuda,'o-',label='CUDA')
    plt.plot(2*sizes,mean_time_cublas,'o-',label='cuBLAS')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Run time (sec.)')
    plt.legend()

    plt.show()







