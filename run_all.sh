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
start='202005190000' # Format YYYYMMDDHHMM or 'realtime'
end='202005200000' # Format YYYYMMDDHHMM or 'realtime'
#station_id=48650 #98222 # TODO : Remove the dependence on this in plot_synop
bbox='99,0.5,106,7.5' # xmin, ymin, xmax, ymax
event_location_name='Johor' # A short name to decribe the location of the event
event_region_name='PeninsulaMalaysia' # This should be a large region for which you can group events together (e.g. Luzon, Java, Terrengganu)

jobs='extractUM downloadGPM downloadUM plot_tephi' # Which scripts to run? Possible values:
# extractUM downloadGPM downloadUM plot_tephi plot_precip plot_walkercirculation plot_synop make_summary_html

######################################################################################################################

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
      python extractUM.py ${start} ${end} ${bbox} ${eventname}
    fi

    # Download GPM IMERG data
    if [ ${j} == 'downloadGPM' ]; then
      python downloadGPM.py auto ${start} ${end} ${organisation}
    fi

    # Download UM data
    if [ ${j} == 'downloadUM' ]; then
      python downloadUM.py ${start} ${end} ${bbox} ${eventname} ${organisation}
    fi

    # Plot Precipitation data
    ## Includes: GPM animations, model vs GPM, and GPM+Analysis Winds vs Model(precip+winds)
    if [ ${j} == 'plot_precip' ]; then
      python plot_precip.py ${start} ${end} ${eventname} ${event_location_name} ${bbox} ${organisation}
    fi

    # Plot postage stamps of GPM vs models
#    if [ ${j} == 'plot_timelagged' ]; then
#      # TODO : merge this script into plot_precip
#      # python plot_timelagged.py ${start} ${end} ${bbox} ${eventname} ${organisation}
#    fi

    # Plot Walker Circulation
    if [ ${j} == 'plot_walkercirculation' ]; then
      python plot_walkercirculation.py ${start} ${end} 'analysis' ${eventname} ${organisation}
    fi

    # Plot Upper Air soundings for each organisation vs models (includes download)
    if [ ${j} == 'plot_tephi' ]; then
      python plot_tephi.py ${start} ${end} ${bbox} ${eventname} ${organisation}
    fi

    # Plot SYNOP data from each organisation vs models
    if [ ${j} == 'plot_synop' ]; then
      python plot_synop.py ${organisation} ${start} ${end} # ${station_id} # Note: station_id is optional
    fi

    # Make an html page summarising all of the output plots
#    if [ ${j} == 'make_summary_html' ]; then
#      # TODO use code from plot_timelagged to auto-generate a summary html page
#      python make_summary_html.py ${organisation}
#    fi

done
