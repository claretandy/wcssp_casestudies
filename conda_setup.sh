#!/bin/bash -l

# Check to see if the conda 'scitools' environment exists
test=$(conda info --envs | grep 'scitools' | cut -d' ' -f1)
if [ $test == 'scitools' ]; then

    conda activate scitools

else

    conda create -n scitools python=3.8
    # Now that we have created the scitools environment, install some things ...
    conda install -c conda-forge -n scitools iris
    conda install -c conda-forge -n scitools h5py
    conda install -c conda-forge -n scitools wget # Not sure this is needed now
    conda install -c conda-forge -n scitools PIL # Not sure this is needed now

    conda activate scitools

fi



