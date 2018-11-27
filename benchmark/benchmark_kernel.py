from iScore.graphrank.graph import Graph
from iScore.graphrank.kernel import Kernel
import networkx as nx
from networkx.algorithms import bipartite

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

    test_cpu(G1,G2)
    #test_gpu(G1,G2)






