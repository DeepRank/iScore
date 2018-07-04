from iScore.graph import GenGraph, Graph

pdb = './input/1ATN.pdb'
pssm = {'A':'./input/1ATN.A.pdb.pssm','B':'./input/1ATN.B.pdb.pssm'}

g = GenGraph(pdb,pssm)
g.construct_graph()
g.export_graph('1ATN.pkl')

g = Graph('1ATN.pkl')
gcheck = Graph('1ATN.mat')
g.compare(gcheck)





