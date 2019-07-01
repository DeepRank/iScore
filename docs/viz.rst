Visualizing the connection graphs
======================================

iScore allows to easily visualize the bipartite graphs using the HDF5 browser provided by the software and pymol. First, the bipartite graphs must be stored in the format of a HDF5 file. To do so, the graphs can be processed to fit in HDF5 file format as follows:


>>> from iScore.graphrank.graph import iscore_graph
>>> iscore_graph(pdb_path=<pdb_path>,
>>>              pssm_path=<pssm_path>,
>>>              export_hdf5=True)

where you have to specify the folder containing the PDB files and PSSM files in ``pdb_path`` and ``pssm_path``. By default, these are set as ``./pdb/`` and ``./pssm/``. The script above creates a HDF5 file containing the graphs.

The generated HDF5 file can be opened using the HDF5 browser. To open the HDF5 file in the HDF5 browser, please go to the ``./h5x/`` folder and type:

``./h5x.py``

You can open a HDF5 file by clicking on the file icon in the bottom left of the browser. Once it is opened, you can see the content of the file in the browser. Right-click on the name of a conformation and choose ``3D Plot``. This will open PyMol and allow you to visualize the bipartite graph

.. image :: h5x_iscore.png
