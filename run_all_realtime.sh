#!/bin/bash -l

# This script runs all the plotting functions for model evaluation

# Either setup the conda environment or activate it if it already exists
#. run_setup.sh
conda activate scitools

######################################################################################################################
# Change things in here for each case study

export organisation='UKMO' # Can be  PAGASA, BMKG, MMD, UKMO or Andy-MacBook. Anything else defaults to 'generic'
export start='realtime' # Format YYYYMMDDHHMM or 'realtime'
export end='realtime' # Format YYYYMMDDHHMM or 'realtime'
#station_id=48650 #98222 # TODO : Remove the dependence on this in plot_synop
export bbox='100,0,110,10' # xmin, ymin, xmax, ymax
export event_location_name='Peninsula-Malaysia' # A short name to decribe the location of the event
export event_region_name='SEAsia' # This should be a large region for which you can group events together (e.g. Luzon, Java, Terrengganu)

# Which scripts to run? Space separated names of scripts
jobs=( plot_precip plot_tephi plot_walkercirculation )

# Possible values:
# extractUM: UKMO only! Extracts UM data from our archive
# downloadGPM: Downloads GPM data into daily netcdf files, and puts them into a standard dir structure
# downloadUM: Downloads UM data from the UKMO FTP site
# plot_tephi: Plots and downloads sounding data from Wyoming and compares UKMO analysis and model data
# plot_precip: Plots a number of model vs precip plots
# plot_walkercirculation: Plots large scale circulation along the tropics
# plot_synop: Plots local synoptic observations. TODO: access GTS data from ogimet and plot vs model
# make_summary_html: Makes a summary page for the event. TODO: write code to take highlight images from each plot type

######################################################################################################################

# This gets the location of your code, while avoiding sharing your local username and paths via github
# The variable $SLURM_JOB_ID is for running the code within the UK Met Office
if [ -z $SLURM_JOB_ID ] ; then
code_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
else
code_dir="$( dirname "$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')" )"
fi
echo $code_dir
#cd $code_dir

# Set the eventname automatically so it is a standard format of region/date_eventlocation
thisdate=$(echo ${end} | awk '{print substr($0,0,8)}')
if [ ${thisdate} == 'realtime' ]; then
  export eventname='monitoring/realtime_'${event_location_name}
else
  export eventname=${event_region_name}'/'${thisdate}'_'${event_location_name}
fi

# Run all the commands requested
for j in "${jobs[@]}"; do
  echo ${j}
  if [ $organisation == 'UKMO' ]; then
    cat <<EOF > ${code_dir}/batch_output/wcssp_casestudies_tmp_${j}.sh
#!/bin/bash -l
#SBATCH --qos=long
#SBATCH --mem=20000
#SBATCH --ntasks=2
#SBATCH --output=batch_output/wcssp_casestudies_tmp_${j}_%j_%N.out
#SBATCH --time=4320

conda activate scitools

echo ${j}
cd ${code_dir}
python ${code_dir}/${j}.py

EOF

    echo "Running: batch_output/wcssp_casestudies_tmp_${j}.sh"
    sbatch ${code_dir}/batch_output/wcssp_casestudies_tmp_${j}.sh
    sleep 10

  else

    echo ${j}
    python ${j}.py

  fi

done

