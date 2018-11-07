# iScore

**Support Vector Machine on Graph kernel for protein-protein conformation ranking**

[![Build Status](https://secure.travis-ci.org/DeepRank/iScore.svg?branch=master)](https://travis-ci.org/DeepRank/iScore)
[![Documentation Status](https://readthedocs.org/projects/iscoredoc/badge/?version=latest)](http://iscoredoc.readthedocs.io/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/DeepRank/iScore/badge.svg?branch=master)](https://coveralls.io/github/DeepRank/iScore?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9491c221796e49c0a120ada9aed5fe42)](https://www.codacy.com/app/NicoRenaud/iScore?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=DeepRank/iScore&amp;utm_campaign=Badge_Grade)

<https://iscoredoc/readthedocs.io/en/latest/>

iScore offers simple solutions to classify protein-protein interfaces using a support vector machine approach on graph kernels. The simplest way to use iScore is through dedicated binaries that hide the complexity of the approach and allows access to the code with simple command line interfaces. The two binaries are `iscore.train` and `iscore.predict` that respectively train a model using a trainging set and use this model to predict the near-native character of unkown conformations.

To use these binaries easily the following file structure is advised :

```
root/
 |__train/
 |    |__ pdb/
 |    |__ pssm/
 |    |__ caseID.lst
 |__predict/
      |__pdb/
      |__pssm/
      |__ caseID.lst (optional)
```

The `train` subdirectory contains the PDB files and the PSSM files of the conformation contained in the training set. The PSSM files can be calculated using PSSMGen <https://github.com/DeepRank/PSSMGen>. To train the model simply go to the `train` subdirectory and type:

```
iScore.train
```

This binary will generate a archive file called by default `training_set.tar.gz` that contains all the information needed to predict binary classes of a test set using the trained model. To use this model go into the `test` subdirectory and type:

```
iScore.predict --archive ../train/training_set.tar.gz
```

This binary will output the binary class and decision value of the conformations in the test set in a text file `iScorePredict.txt`.

## MPI Binaries

MPI binaries have also been developped to facilitate the processing of large training and test dataset. These binaries can simply be used as:

```
mpiexec -n ${NPROC} iScore.train.mpi
mpiexec -n ${NPROC} iScore.predcit.mpi
```

## Graph Kernel on GPUs

CUDA enables routines have been implemented to accelerate the calculation og the graph kernels using GPUs. This feature os not accessible directly via the `iScore.train` and `iScore.predict` binaries but can only be used through lower-level binaries shipped by iScore. Go to the official iScore documentaion to learn how to access these features <https://iscoredoc/readthedocs.io/en/latest/>