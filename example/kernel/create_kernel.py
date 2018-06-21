#!/usr/bin/env python
from iScore.graph import Graph, iscore_graph
from iScore.kernel import Kernel

# create all the graphs of the pdb in ./pdb/
#iscore_graph()


# g = Graph('./graph/2OZA.pckl')
# gcheck = Graph('./check/graphMAT/2OZA.mat')
# g.compare(gcheck)


# g = Graph('./graph/1IRA.pckl')
# gcheck = Graph('./check/graphMAT/1IRA.mat')
# g.compare(gcheck)


#init and load the data
ker = Kernel()
ker.import_from_mat()

# get the path of the check file
#checkfile = ker.get_check_file(fname=None)

# run the calculations
ker.run(lamb=1.0,walk=4,check=None)