Installation
==============================

The code is hosted on Github_ (https://github.com/DeepRank/iScore)

.. _Github: https://github.com/DeepRank/iScore

To install the code

 * clone the repository ``git clone https://github.com/DeepRank/iScore.git``
 * go there ``cd iScore``
 * install the module ``pip install -e ./``

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

  * mpi4py

  * libsvm (https://github.com/cjlin1/libsvm/tree/master/python)

  * pdb2sql(https://github.com/DeepRank/pdb2sql)
