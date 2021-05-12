#!/bin/bash -l

code_dir=$HOME/GitHub/wcssp_casestudies
cd $code_dir

conda activate scitools

python ${code_dir}/run_extract.py
