Graphs and Kernels
===============================


Generating the Graphs :
----------------------------


The first step in iScore is to generate the bipartite graph of the interface of a decoy model. In the generated graph, each node is encoded the PSSM profile of a residue. The nodes are connected if they form a contact pair between two proteins of the decoy model.

To create the graph, a PDB file for the interface and two separate PSSM files (one for each protein chain) created by the PSSMGen tool are required. To generate the graph, simply use :

>>> from iScore.graph import GenGraph, Graph
>>> 
>>> pdb = name.pdb
>>> pssm = {'A':'name.A.pdb.pssm','B':'name.B.pdb.pssm'}
>>> 
>>> g = GenGraph(pdb,pssm)
>>> g.construct_graph()
>>> g.export_graph('name.pkl')

This simple example constructs the bipartite graph and export it into a pickle file. A working example can be found in ``example/graph/create_graph.py``

The function ``iscore_graph()`` facilitates generation of a large number of conformations. By default, this function creates the graphs of all conformations stored in the subfolder ``./pdb/`` using the PSSM files stored in the subfolder ``./pssm/``. The resulting graphs will be stored in the subfolder ``./graph/``.

Generating the Graph Kernels :
-------------------------------------

Once we obtain the graphs of conformations, we can simply compute the kernel of the different pairs using iScore. An example can be found at ``example/kernel/create_kernel.py``

>>> from iScore.graph import Graph, iscore_graph
>>> from iScore.kernel import Kernel
>>> 
>>> # create all the graphs of the pdb in ./pdb/
>>> iscore_graph()
>>> 
>>> #init and load the data
>>> ker = Kernel()
>>> ker.import_from_mat()
>>> 
>>> # run the calculations
>>> ker.run(lamb=1.0,walk=4,check=checkfile)

The kernel between the two graphs is computed by the `Kernel()` class. By default, the method `Kernel.import_from_mat()` imports all the graphs stored in the subfolder `graph/`. To compute all pairwise kernels of the graphs loaded, we can simply use the method `Kernel.run()` (Yong: which one is correct, ker.run or Kernel.run?) . Users can set a lambda value and a walking length as parameters.
