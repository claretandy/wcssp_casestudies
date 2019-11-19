

:: Runs all the plotting functions for model evaluation

:: Load the conda environment
:: TODO : work out a clever way to pick up what the locally preferred environment is (e.g. Andy's env is called scitools, but PAGASA's is called 'nms)
call conda activate scitools
::conda activate nms

:: If you haven't already, you might need to install the following packages
::conda install -c conda-forge h5py
::conda install -c conda-forge wget

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:: Change things in here for each case study
set organisation=BMKG
:: Can be  PAGASA, BMKG, MMD, UKMO or Andy-MacBook. Anything else defaults to 'generic'
set start=201911030000
:: Format YYYYMMDDHHMM
set end=201911040000
:: Format YYYYMMDDHHMM
set station_id=98222
:: TODO : Georeference each station ID so that they can be selected using a spatial query
set event_domain=105,-7,108,-5
:: xmin, ymin, xmax, ymax
set event_location_name=Jakarta
:: A short name to decribe the location of the event
set event_region_name=Java
:: This should be a large region for which you can group events together (e.g. Luzon, Java, Terrengganu)
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:: Set the eventname automatically so it is a standard format of region/date_eventlocation
set eventname=Java/20191119_Jakarta
::%event_location_name%
::'/'$(echo ${end} | awk '{print substr($0,0,8)}')'_'${event_location_name}

:: Run scripts to plot case study data
:: Download GPM IMERG data
::python downloadGPM.py auto %start% %end% %organisation%

:: Plot GPM animation for different time aggregations
python nrt_plots_v3_casestudies.py %start% %end% %event_domain% %eventname%
:: TODO : make this script work in this environment

:: Get UM model data from FTP
::python downloadUM.py %organisation%
:: TODO : Either download from UKMO ftp site, or find files locally

:: Plot postage stamps of GPM vs models
::python plot_timelagged.py %start% %end% %event_domain% %eventname% %organisation%
:: TODO : make this script work in this environment - could also be adapted for other satellite obs / analysis

:: Plot SYNOP data from each organisation vs models
::python plot_synop.py ${organisation} ${start} ${end} ${station_id}
:: Note: station_id is optional

:: Plot Upper Air soundings for each organisation vs models
::python plot_tephi.py

:: Make an html page summarising all of the output plots
::python make_html.py ${organisation}
:: TODO use code from plot_timelagged to auto-generate a summary html page


