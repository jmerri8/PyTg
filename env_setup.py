conda create -n pm3env -c conda-forge "python=3.8" libpython mkl-service m2w64-toolchain numba python-graphviz scipy
conda activate pm3env
conda install pymc3
conda install spyder=4.2.5
conda remove theano
conda install -c conda-forge theano-pymc
