.. highlight:: rst

Introduction
=============================

**Support Vector Machine on Graph Kernels for Protein-Protein Docking Scoring**

iScore uses support vector machine to classify protein-protein interfaces in enar natives and non-near natives. To this purpose, each interface is encoded in a graph where each node corresponds to the Position Specific Similarity Matrix (PSSM) of a given residue. The nodes can be connected if they participate in a contact between the two different proteins.

The graph of a given conformation is then used to computeits kernel with a training set of carefully chosen conforations representing near native and decoys. A support vector machine is then adopted to predict if the conformation under scrutiny is near native or not.

Installation
-------------------------------

The code is hosted on Github_ (https://github.com/DeepRank/iScore)

.. _Github: https://github.com/DeepRank/iScore

To install the code

 * clone the repository ``git clone https://github.com/DeepRank/iScore.git``
 * go there ``cd iScore``
 * install the module ``pip insall -e ./``

Test the installation
----------------------

To test the module go to the test folder ``cd ./test`` and execute the following test : ``pytest``

These tests are automatically run on Travis CI at each new push.
So if the build button display passing they should work !

Requiried Dependencies
------------------------

The code is written in Python3. Several packages are required to run the code but most are pretty standard. Here is an non-exhaustive list of dependencies

  * Numpy

  * Biopython

  * libsvm

  * mpi4py

  * pdb2sql




