#!/bin/bash -l

test=$(conda info --envs | grep 'scitools' | cut -d' ' -f1)
if [ $test == 'scitools' ]; then

  conda activate scitools

else

  # Create a new conda environment
  conda create -n scitools python=3.8

  # Install some important packages
  conda install -c conda-forge -n scitools iris
  conda install -c conda-forge -n scitools h5py
  conda install -c conda-forge -n scitools wget # Possibly not required anymore
  conda install -c conda-forge -n scitools PIL # Possibly not required anymore

  conda activate scitools

  # Install the latest version of tephi for tephigram plotting
  thisdir=$(pwd)
  cd ../
  git clone https://github.com/SciTools/tephi.git
  cd tephi
  python setup.py install
  cd ${thisdir}

fi
