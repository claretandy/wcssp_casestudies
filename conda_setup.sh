#!/bin/bash -l

# Check to see if the conda 'scitools' environment exists
test=$(conda info --envs | grep 'scitools' | cut -d' ' -f1)
if [ $test == 'scitools' ]; then
  conda activate scitools
else
  conda create -n scitools
fi

# Now that we have either created it or activated it,
conda install -c conda-forge h5py
conda install -c conda-forge wget
conda install -c conda-forge PIL
conda install -c conda-forge ftplib
conda install -c conda-forge lxml
