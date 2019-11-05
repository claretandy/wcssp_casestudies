#!/bin/bash -l
#SBATCH --mem=20000
#SBATCH --ntasks=2
#SBATCH --output=/scratch/hadhy/tafrica/batch_out/plotTimelagged_slurm_%j_%N.out
#SBATCH --time=360 #4320
#SBATCH --qos=high #long

module load scitools/experimental-current
timeagg=( 3 6 12 24 )
lvb_domain='22,-12,52,15'
plotscript=/home/h02/hadhy/Repository/hadhy_scripts/WCSSP/functions/plot_timelagged.py
searchlist='ga6,km4p4'

for ta in ${timeagg[@]}; do

    python ${plotscript} 201903060000 201903070000 ${ta} ${lvb_domain} ${searchlist} 20190306_LakeVictoria # For 12 or 24 hours

done
