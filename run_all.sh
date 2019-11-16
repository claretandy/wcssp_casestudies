#!/bin/bash -l

# Runs all the plotting functions for model evaluation

conda activate scitools

organisation='Andy-MacBook' # Can be
start='201911010000'
end='201911080000'

# Script locations relative to this script
synop_script='plot_synop.py'
plot_tephi_script='plot_tephi.py'

python ${synop_script} ${organisation} ${start_dt} ${end_dt} ${station_id}
python ${plot_tephi_script}

