#!/bin/bash -l
#SBATCH --qos=high
#SBATCH --mem=10000
#SBATCH --ntasks=2
#SBATCH --output=/scratch/hadhy/batch/wcssp_casestudies/runall_slurm_%j_%N.out
#SBATCH --time=360

# This script runs all the plotting functions for model evaluation

# Either setup the conda environment or activate it if it already exists
#. run_setup.sh
conda activate scitools

######################################################################################################################
# Change things in here for each case study

organisation='UKMO' # Can be  PAGASA, BMKG, MMD, UKMO or Andy-MacBook. Anything else defaults to 'generic'
start='realtime' # Format YYYYMMDDHHMM or 'realtime'
end='realtime' # Format YYYYMMDDHHMM or 'realtime'
#station_id=48650 #98222 # TODO : Remove the dependence on this in plot_synop
bbox='100,0,110,10' # xmin, ymin, xmax, ymax
event_location_name='Peninsula-Malaysia' # A short name to decribe the location of the event
event_region_name='SEAsia' # This should be a large region for which you can group events together (e.g. Luzon, Java, Terrengganu)

jobs='plot_precip plot_tephi plot_walkercirculation' # Which scripts to run? Possible values:
# extractUM downloadGPM downloadUM plot_tephi plot_precip plot_walkercirculation plot_synop make_summary_html

######################################################################################################################

# This reads the .config file to get the location of your code
# Also, it avoids sharing your local username and paths
#if [ -n $SLURM_JOB_ID ] ; then
#code_dir="$( dirname "$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')" )"
#else
#code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#fi
code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $code_dir


#code_dir=$(grep $organisation .config -A 13 | grep code_dir | cut -d= -f2)
#cd $code_dir

# Set the eventname automatically so it is a standard format of region/date_eventlocation
thisdate=$(echo ${end} | awk '{print substr($0,0,8)}')
if [ ${thisdate} == 'realtime' ]; then
  eventname='monitoring/realtime_'${event_location_name}
else
  eventname=${event_region_name}'/'${thisdate}'_'${event_location_name}
fi

for j in ${jobs[@]}; do

    # If running from inside the Met Office, extract data for this case study and share on FTP
    if [ ${organisation} == 'UKMO' ] && [ ${j} == 'extractUM' ]; then
      python ${code_dir}/extractUM.py ${start} ${end} ${bbox} ${eventname}
    fi

    # If running from a different organisation, download UM data from FTP
    if [ ${j} == 'downloadUM' ] && [ ${organisation} != 'UKMO' ]; then
      python ${code_dir}/downloadUM.py ${start} ${end} ${bbox} ${eventname} ${organisation}
    fi

    # Download GPM IMERG data
    if [ ${j} == 'downloadGPM' ]; then
      python ${code_dir}/downloadGPM.py auto ${start} ${end} ${organisation}
    fi

    # Plot Precipitation data
    ## Includes: GPM animations, model vs GPM, and GPM+Analysis Winds vs Model(precip+winds)
    if [ ${j} == 'plot_precip' ]; then
      python ${code_dir}/plot_precip.py ${start} ${end} ${eventname} ${event_location_name} ${bbox} ${organisation}
    fi

    # Plot Upper Air soundings for each organisation vs models (includes download)
    if [ ${j} == 'plot_tephi' ]; then
      python ${code_dir}/plot_tephi.py ${start} ${end} ${bbox} ${eventname} ${organisation}
    fi

    # Plot Walker Circulation
    if [ ${j} == 'plot_walkercirculation' ]; then
      python ${code_dir}/plot_walkercirculation.py ${start} ${end} 'analysis' ${eventname} ${organisation}
    fi

    # Plot SYNOP data from each organisation vs models
    if [ ${j} == 'plot_synop' ]; then
      python ${code_dir}/plot_synop.py ${organisation} ${start} ${end} # ${station_id} # Note: station_id is optional
    fi

    # Make an html page summarising all of the output plots
#    if [ ${j} == 'make_summary_html' ]; then
#      # TODO use code from plot_timelagged to auto-generate a summary html page
#      python make_summary_html.py ${organisation}
#    fi

done
