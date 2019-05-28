Quick example
==================================

iScore is most easily used via a set of binaries that are located in the directory `iscore/bin/`. Thes binaries allows to create graphs, compute kernel traina SVM model and use it in just a few command lines. The only restriction is that they have to be called in a well defined directory structure.

We therefore recomend to have the following file arborescence:

::
root/
 |__train/
 |    |__pdb/
 |    |__pssm/
 |    |__caseID.lst
 |__predict/
      |__pdb/
      |__pssm/
      |__caseID.lst (optional)
::

The subfolders are ``train/pdb/`` and ``train/pssm/`` contains the pdb and pssm files we want to use to create a training set. The PSSM files can be obtained via a dedicated Python module `PSSMGen` (see section 'Computing PSSM Files'). To train a SVM model on these conformation simply go in the ``train`` subfolder and use the `iScore.train` binary:


  * ``cd root/train/``
  * ``mpiexec -n 4 iScore.train.mpi``


This binary will automatically generates the graphs of the interfaces, compute their pairwise kernels and train a SVM model using those kernels. The binary will produce an archive file called ``training_set.tar.gz`` that can be used to predict the near native character of the new conformations.

To do that go in the ``root/predict/`` subfolder and use the ``iSore.predcit`` binary:


  * ``cd root/train/``
  * ``mpiexec -n 4 iScore.predict.mpi --archive ../train/training_set.tar.gz``

This will use the training set to evaluate the near-native character of all the conformations present in the test set. The result will be outputed in a text file ``iSorePredict.dat``.