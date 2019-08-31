# PtychoCG
Conjugate gradient solver for 2d ptychography with probe retrieval

## Installation from source
export CUDAHOME=path-to-cuda

python setup.py install

## Dependency 
cupy - for GPU acceleration of linear algebra operations in iterative schemes. See (https://cupy.chainer.org/). For installation use

conda install -c anaconda cupy

## Tests
Run python test/test.py