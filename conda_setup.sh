#!/bin/bash -l

# Check to see if the conda 'scitools' environment exists
test=$(conda info --envs | grep 'scitools' | cut -d' ' -f1)
if [ $test == 'scitools' ]; then
  conda activate scitools
else
  conda create -n scitools python=3.8
fi

conda activate scitools

# Now that we have either created it or activated it,
conda install -c conda-forge -n scitools h5py
conda install -c conda-forge -n scitools wget
conda install -c conda-forge -n scitools PIL
