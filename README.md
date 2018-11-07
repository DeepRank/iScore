# iScore

**Graph kernel approach of  conformation ranking**

https://iscoredoc/readthedocs.io/en/latest/

[![Build Status](https://secure.travis-ci.org/DeepRank/iScore.svg?branch=master)](https://travis-ci.org/DeepRank/iScore) 
[![Documentation Status](https://readthedocs.org/projects/iscoredoc/badge/?version=latest)](http://iscoredoc.readthedocs.io/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/DeepRank/iScore/badge.svg?branch=master)](https://coveralls.io/github/DeepRank/iScore?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9491c221796e49c0a120ada9aed5fe42)](https://www.codacy.com/app/NicoRenaud/iScore?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=DeepRank/iScore&amp;utm_campaign=Badge_Grade)


## Install

You need:
  * libsvm  (https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
  * pdb2sql (https://github.com/DeepRank/pdb2sql)
  * mpi4py (pip install mpi4py)

The python binding of libsvm may not add the path to the python path. Don't forget to add `export: PYTHONPATH="/path/to/libsvm/python:$PYTHONPATH"` to your .bashrc. Install iScore with

`pip install -e ./`

as usual. Some executables are stored in `iScore/bin/`. Don't forget to add this to your bashrc as well: `export PATH="/path/to/iScore/bin:$PATH"`

## Test

To test the install go to `test` and run the test suite by typing

```
pytest
```

## Example

A few examples are given in the folder iScore/example.

### Computing graphs

The file `iScore/example/graph/create_graph.py` shows how to generate the graph of a given conformation specified by the PDB and PSSM file located in the subfolder input/

```python
from iScore.graph import GenGraph, Graph

pdb = './input/1ATN.pdb'
pssm = {'A':'./input/1ATN.A.pdb.pssm','B':'./input/1ATN.B.pdb.pssm'}

g = GenGraph(pdb,pssm)
g.construct_graph()
g.export_graph('1ATN.pkl')

g = Graph('1ATN.pkl')
gcheck = Graph('1ATN.mat')
g.compare(gcheck)
```

To generate graph the class `GenGraph` can be used with the path of the pdb file and pssm files given as a dictionary. The method `GenGraph.construct_graph()` will then assemble the graph that one can export with the method `Graph.export_graph()`.

You can also read an existing graph with the class 'Graph'. Two `Graph` instance can be comapred with the metod 'Graph.compare()'.

### Computing Kernels

The file `iScore/example/kernel/create_kernel.py` shows how to compute the kernel of two graphs


```python
from iScore.graph import Graph, iscore_graph
from iScore.kernel import Kernel

# create all the graphs of the pdb in ./pdb/
iscore_graph()

# comapre the graphs with the ones obtained
# with the matlab version
g = Graph('./graph/2OZA.pckl')
gcheck = Graph('./check/graphMAT/2OZA.mat')
g.compare(gcheck)


g = Graph('./graph/1IRA.pckl')
gcheck = Graph('./check/graphMAT/1IRA.mat')
g.compare(gcheck)


#init and load the data
ker = Kernel()
ker.import_from_mat()

# get the path of the check file
checkfile = ker.get_check_file(fname='./check/kernelMAT/K.mat')

# run the calculations
ker.run(lamb=1.0,walk=4,check=checkfile)
```

Before computing the kernel the graphs of the two conformation stored in `pdb/` are calculated. This is here done with a single command `iscore_graph()`. By default this function will create the graphs of all the conformations stored in the subfolder `pdb/` with the pssm stored in the subfolder `pssm/`. We here also check that the graphs are identical to the ones stored in the `check/graphMAT/` folder.

The kernel between the two graphs computed above is calculated with the class `Kernel()`. By default the method `Kernel.import_from_mat()` will read all the graphs stored in the subfolder `graph/`. We also check here that the kernel obtained by the `Kernel` instance are identical to the ones stored in 'check/kernelMAT/'. To compute all the pairwise kernels of the graphs loaded above we can simply use the method `Kernel.run()`. We can here specify the value of lambda and the length of the walk.


### Workflow with the iScore binaries

iScore also comes with binaries that can be used directly from the command line. To illustrate the use of these libraries go to the folder `iScore/example/training_set/`. The subfolders are `pdb/` and `pssm/` contains here the pdb and pssm files we want to use to create a training set and model trained on it. The binary class corresponding to these conformations are specified in the file 'caseID.lst' This operation can here be done in one single step :

```
iScore.train
```

This binary will first generate the graphs of the conformations stored in `pdb/` and `pssm/`. These graphs will be stored in `graph/`. The binary  will then compute the pairwise kernels of these graphs and store the kernel files in `kernel/`. Finally the binary will train a SVM model using the kernel files and the `caseID.lst` file that contains the binary class of the model.

The calculated graphs and the svm model are then stored in a single tar file called here `training_set.tar.gz`. This file therefore contains all the information needed to predict binary classes of a test set using the trained model.

To predict binary classes (and decision values) of new conformation go to the subfoler `test/`. Here 5 conformations are specified by the pdb and pssm files stored in `pdb/` and `pssm/` that we want to use as a test set and predict their binary class. This can be achieve with the binary `iScore.predict` following:

```
iScore.predict --archive ../training_set.tar.gz
```

This command will use first compute the graph of the comformation in the test set and store them in `graph/`. The binary will then compute the pair wise kernels of each graph in the test set with all the graph contained in the training set that are stored in the tar file. These kernels will be stored in `kernel/`. Finally the binary will use the trained SVM model contained in the tar file to predict the binary class and decision value of the conformations in the test set. The results are then stored in a text file and a pickle file `iScorePredict.pkl` and `iScorePredict.txt`. Opening the text file you will see :

```
Name       label      pred     decision_value
1ACB_2w     None         0             -0.994
1ACB_3w     None         0             -0.994
1ACB_1w     None         0             -0.994
1ACB_4w     None         0             -0.994
1ACB_5w     None         0             -0.994

```

The ground truth label are here all None because they were not provided in the test set. This can simply be done by adding a `caseID.lst` in the `test/` subfolder.


### MPI Binaries

MPI enabled binaries allows to split the calculation of the graph and kernel over multiple core easily. You need `mpi4py` installed. *At the moment it is impossible to use simultaneously the MPI execs and the CUDA kernels.* If you use the MPI binaries all the kernels will be calculated on CPUs only.


 We can illustrate their use with the example in `iScore/example/training_set/` already  used above. To create the training set using mpi simply go in that directory and type :

```
mpiexec -n 2 iScore.train.mpi
```

This command will use two core to compute the training set (graphs, kernel, svm). Similarly to use the archive file generated here go to the `test` subfolder and type :

```
mpiexec -n 2 iScore.predcit.mpi
```

This command will compute the graphs and their pair wise kernel witht the training set using 2 cores.


Of course if you're imteding to run that on a large number of core on a cluster write a small bash script like:

```
#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -N svm_train

mpiexec -n 48 iScore.train.mpi
```

and submit that job the queue.
