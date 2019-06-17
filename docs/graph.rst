Graphs and Kernels
===============================


Generating the Graphs :
----------------------------


The first step in iScore is to generate the bipartite graph of the interface. In this graph each node is represented by the PSSM profile of a residue. The nodes are connected if they form a contact pair between the two proteins.

To create the graph one needs the PDB file of the interface and the two PSSM files (one for each chain) created by the PSSMGen tool. To generate the graph simply use :

>>> from iScore.graph import GenGraph, Graph
>>> 
>>> pdb = name.pdb
>>> pssm = {'A':'name.A.pdb.pssm','B':'name.B.pdb.pssm'}
>>> 
>>> g = GenGraph(pdb,pssm)
>>> g.construct_graph()
>>> g.export_graph('name.pkl')

This simple example will construct the connection graph and export it in a pickle file. A working example can be found in ``example/graph/create_graph.py``

The function ``iscore_graph()`` facilitate the generation of a large number of conformations. By default this function will create the graphs of all the conformations stored in the subfolder ``./pdb/`` using the pssm files stored in the subfolder ``./pssm/``. The resulting graphs will be stored in the subfolder ``./graph/``.

Generating the Graph Kernels :
-------------------------------------

Once we have calculated the graphs of multiple conformation we can simply compute the kernel of the different pairs using iScore. An example can be found at ``example/kernel/create_kernel.py``

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

The kernel between the two graphs computed above is calculated with the class `Kernel()`. By default the method `Kernel.import_from_mat()` will read all the graphs stored in the subfolder `graph/`. To compute all the pairwise kernels of the graphs loaded above we can simply use the method `Kernel.run()`. We can here specify the value of lambda and the length of the walk.
