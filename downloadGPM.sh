#!/bin/bash -l

conda activate nms

today=$(date -u +%Y%m%d)
outdir=''
agency='pagasa'

python downloadGPM.py NRTearly 7 ${today} ${outdir} ${agency}