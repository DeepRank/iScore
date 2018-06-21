# iScore

Graph kernel approach of  conformation ranking

## Install

You need:

  . libsvm
  . pdb2sql

The python bonding of libsvm may not add the path to the python path. Don't forget to add

`export: PYTHONPATH="/path/to/libsvm/python:$PYTHONPATH"`

to your .bashrc. Install iScore with

`pip install -e ./`

as usual. Some executables are stored in `iScore/bin/`. Don't forget to add this to your bashrc as well
`export PATH="/path/to/iScore/bin:$PATH"`

## Test

To test the install go to `iScore/example/`. Two simple examples are stored in `./graph/` and './kernel/'. For example in `./graph/` type

`python create_graph.py`

to generate the graphs of a single conformation

In './kernel/' type:

`python create_kernel.py`

to generate the graphs and compute the kernels of two coformations.

In `./training_example` a simple workflow can be executed. As you can notive there is a bin folder. To generate the training set simply type:

`iScore.train`

This will generate the graphs of the conformations of this training set. Then it will compute the kernels and finally train a SVM model. The final result is stored in a tar file calles `training_set.tar.gz` that contains all the information needed to predict values from the model. 

To use the model go to the subfoler `./test/`. There 5 coformations are present and can be tested against the model trained above. To do that simply type:

`iScore.predict --archive ../training_set.tar.gz`

This wil compute the graphs of the test set, compute the kernels with the grpahs of the trainig set and predict the classes from the trained SVM model.


More Documentation to come soon

