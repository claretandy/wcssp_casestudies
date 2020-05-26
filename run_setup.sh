#!/bin/bash -l

# Create a new conda environment
conda create -n scitools

# Install some important packages
conda install -c conda-forge iris
conda install -c conda-forge h5py
conda install -c conda-forge wget # Possibly not required anymore
conda install -c conda-forge PIL # Possibly not required anymore

# Install the latest version of tephi for tephigram plotting
thisdir=$(pwd)
cd ../
git clone https://github.com/SciTools/tephi.git
cd tephi
python setup.py install
cd ${thisdir}
