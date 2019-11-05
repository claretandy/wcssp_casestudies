#!/bin/bash -l
#SBATCH --mem=20000
#SBATCH --ntasks=2
#SBATCH --qos=high
#SBATCH --output=/scratch/hadhy/CaseStudies/casestudies_slurm_%j_%N.out
#SBATCH --time=360

module load scitools

# GPM case studies
gpm_casestudy_script='/home/h02/hadhy/Repository/hadhy_scripts/WCSSP/functions/nrt_plots_v3_casestudies.py'

python ${gpm_casestudy_script} 20171230 20180104 99.65,1,106,7.45 20180103_PeninsulaMalaysia
python ${gpm_casestudy_script} 20180111 20180114 99.65,1,106,7.45 20180113_PeninsulaMalaysia
python ${gpm_casestudy_script} 20180112 20180116 116,5,129,18 20180115_Philippines-Visayas
python ${gpm_casestudy_script} 20180203 20180207 105.5,-3.5,115.5,6.5 20180206_Borneo
python ${gpm_casestudy_script} 20180203 20180207 104,-8,112,0 20180206_Jakarta

