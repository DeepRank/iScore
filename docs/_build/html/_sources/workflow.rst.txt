iScore Workflow
========================

Serial binaries
-----------------------

iScore also comes with binaries that can be used directly from the command line. To illustrate the use of these libraries go to the folder ``iScore/example/training_set/``. The subfolders are ``pdb/`` and ``pssm/`` contains here the pdb and pssm files we want to use to create a training set and model trained on it. The binary class corresponding to these conformations are specified in the file 'caseID.lst' This operation can here be done in one single step :


$ iScore.train


This binary will first generate the graphs of the conformations stored in ``pdb/`` and ``pssm/``. These graphs will be stored in ``graph/``. The binary  will then compute the pairwise kernels of these graphs and store the kernel files in ``kernel/``. Finally the binary will train a SVM model using the kernel files and the ``caseID.lst`` file that contains the binary class of the model.

The calculated graphs and the svm model are then stored in a single tar file called here ``training_set.tar.gz``. This file therefore contains all the information needed to predict binary classes of a test set using the trained model.

To predict binary classes (and decision values) of new conformation go to the subfoler ``test/``. Here 5 conformations are specified by the pdb and pssm files stored in ``pdb/`` and ``pssm/`` that we want to use as a test set and predict their binary class. This can be achieve with the binary ``iScore.predict`` following:


$ iScore.predict --archive ../training_set.tar.gz


This command will use first compute the graph of the comformation in the test set and store them in `graph/`. The binary will then compute the pair wise kernels of each graph in the test set with all the graph contained in the training set that are stored in the tar file. These kernels will be stored in ``kernel/``. Finally the binary will use the trained SVM model contained in the tar file to predict the binary class and decision value of the conformations in the test set. The results are then stored in a text file and a pickle file ``iScorePredict.pkl`` and ``iScorePredict.txt``. Opening the text file you will see :

+--------+--------+---------+-------------------+
|Name    |   label|     pred|     decision_value|
+--------+--------+---------+-------------------+
|1ACB_2w |   None |       0 |           -0.994  |
+--------+--------+---------+-------------------+
|1ACB_3w |   None |       0 |           -0.994  |
+--------+--------+---------+-------------------+
|1ACB_1w |   None |       0 |           -0.994  |
+--------+--------+---------+-------------------+
|1ACB_4w |   None |       0 |           -0.994  |
+--------+--------+---------+-------------------+
|1ACB_5w |   None |       0 |           -0.994  |
+--------+--------+---------+-------------------+


The ground truth label are here all None because they were not provided in the test set. This can simply be done by adding a ``caseID.lst`` in the ``test/`` subfolder.


MPI Binaries
------------------------

MPI enabled binaries allows to split the calculation of the graph and kernel over multiple core easily. You need ``mpi4py`` installed. At the moment it is impossible to use simultaneously the MPI execs and the CUDA kernels. If you use the MPI binaries all the kernels will be calculated on CPUs only.


 We can illustrate their use with the example in ``iScore/example/training_set/`` already  used above. To create the training set using mpi simply go in that directory and type :


$ mpiexec -n 2 iScore.train.mpi


This command will use two core to compute the training set (graphs, kernel, svm). Similarly to use the archive file generated here go to the ``test`` subfolder and type :


$ mpiexec -n 2 iScore.predcit.mpi


This command will compute the graphs and their pair wise kernel witht the training set using 2 cores.


Of course if you're imteding to run that on a large number of core on a cluster write a small bash script like:


#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -N svm_train

mpiexec -n 48 iScore.train.mpi


and submit that job the queue.