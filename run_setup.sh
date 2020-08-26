#!/bin/bash -l

test=$(conda info --envs | grep 'scitools' | cut -d' ' -f1)
if [ $test == 'scitools' ]; then

  conda activate scitools

else

  # Create a new conda environment
  conda create -n scitools python=3.8

  # Install some important packages
  conda install -c conda-forge -n scitools iris
  conda install -c conda-forge -n scitools mo_pack
  conda install -c conda-forge -n scitools h5py
  conda install -c conda-forge -n scitools wget # Possibly not required anymore
  conda install -c conda-forge -n scitools PIL # Possibly not required anymore
  conda install -n scitools sphinx # For documentation
  # conda install -n scitools flask # For creating web pages with jinja2
  # For access to ERA5 data
  # ERA5 access also requires registration at https://cds.climate.copernicus.eu/
  conda install -c conda-forge -n scitools cdsapi

  conda activate scitools

  # Install the latest version of tephi for tephigram plotting
  thisdir=$(pwd)
  cd ../
  git clone https://github.com/SciTools/tephi.git
  cd tephi
  python setup.py install
  cd ${thisdir}

fi
