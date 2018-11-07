#!/bin/bash
set -ex
wget https://github.com/cjlin1/libsvm/archive/v323.tar.gz
tar -xvf v323.tar.gz
cd libsvm-323/python
make
export PYTHONPATH="$PWD:$PYTHONPATH"
cd ../../


