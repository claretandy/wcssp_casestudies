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
start='202005190000' # Format YYYYMMDDHHMM
end='202005200000' # Format YYYYMMDDHHMM
#station_id=48650 #98222 # TODO : Remove the dependence on this
bbox='99,0.5,106,7.5' # xmin, ymin, xmax, ymax
event_location_name='Johor' # A short name to decribe the location of the event
event_region_name='PeninsulaMalaysia' # This should be a large region for which you can group events together (e.g. Luzon, Java, Terrengganu)
######################################################################################################################

# Set the eventname automatically so it is a standard format of region/date_eventlocation
eventname=${event_region_name}'/'$(echo ${end} | awk '{print substr($0,0,8)}')'_'${event_location_name}

# If running from inside the Met Office, extract data for this case study and share on FTP
#if [ $organisation == 'UKMO' ]; then
#  python extractUM.py ${start} ${end} ${bbox} ${eventname}
#fi

# Run scripts to plot case study data
# Download GPM IMERG data
#python downloadGPM.py auto ${start} ${end} ${organisation}

# Plot GPM animation for different time aggregations
#python nrt_plots_v3_casestudies.py 'NRTlate' ${start} ${end} ${bbox} ${eventname} ${organisation}

# Get UM model data from FTP
# Can also be run in realtime as a cronjob:
# python downloadUM.py
#python downloadUM.py ${start} ${end} ${bbox} ${eventname} ${organisation}

# Plot Walker Circulation
python plot_walkercirculation.py ${start} ${end} 'analysis' ${eventname} ${organisation}

## Plot postage stamps of GPM vs models
# TODO : make this script work in this environment - could also be adapted for other satellite obs / analysis
#python plot_timelagged.py ${start} ${end} ${bbox} ${eventname} ${organisation}

## Plot SYNOP data from each organisation vs models
# TODO : remove the dependence on station_id
#python plot_synop.py ${organisation} ${start} ${end} ${station_id} # Note: station_id is optional

## Plot Upper Air soundings for each organisation vs models
python plot_tephi.py ${start} ${end} ${event_domain} ${eventname} ${organisation}

# Make an html page summarising all of the output plots
python make_summary_html.py ${organisation} # TODO use code from plot_timelagged to auto-generate a summary html page

