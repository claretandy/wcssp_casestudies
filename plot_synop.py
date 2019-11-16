#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:36:54 2019

@author: hmiguel
"""
import sys, os
from location_config import load_location_settings
import pandas as pd
import numpy as np
import glob
import datetime as dt
import pdb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



def formatValue(k, v):
    '''
    Formats the values of the SYNOP observations according to the name of the variable.
    Hopefully this will be quite standard between all NMSs
    :param k: The name of the variable
    :param v: The value
    :return: The formatted value
    '''

    # print('k, v: ', k, v)

    if v == 'T':
        return [0]

    if v == '±0':
        v = '0'

    if v == None or v == '':
        return [np.nan]

    if k in ['mslp', 'drybulb', 'wetbulb', 'value', 'observed', 'total', 'temperature']:
        return [float(v)]

    for possk in ['speed', 'direction', 'gust', 'threeHourChange', 'twentyFourHourChange', 'oktas']:
        if possk in k:
            try:
                return [int(float(v))]
            except:
                pdb.set_trace()
            # if k in ['speed', 'direction', 'gust', 'threeHourChange', 'twentyFourHourChange', 'oktas']:
    #     v = int(v)
    #     return [v]

    if 'date' in k:
        try:
            return [dt.datetime.strptime(v, '%Y-%m-%d %H:%M:%S')]
        except:
            return [v]

    if 'time' in k:
        try:
            h,m,s = [int(t) for t in v.split(':')]
            return [dt.time(h, m, s)]
        except:
            return [v]

    return [v]
    
def getData(start_dt, end_dt, settings, station_id=None):
    '''
    start_dt and end_dt are datetime objects
    station_id is optional, but if given, selects just one station
    '''
    
    # Get the paths to the data
    path          = settings['synop_path']
    file_wildcard = settings['synop_wildcard']
    freq          = settings['synop_frequency']
    
    thisdate = start_dt
    outdf = {}

    while thisdate <= end_dt:
        
        thisfile, = glob.glob(path + '*' + thisdate.strftime('%Y%m%d%H%M') + file_wildcard)
        
        df = pd.DataFrame(pd.read_json(thisfile))
        df_subset = df.loc[df['stationNumber'] == station_id]

        if not df_subset.empty:
            this_dict = {}
            for index, val in df_subset.iteritems():

                for v in val:

                    if isinstance(v, list):
                        # If the field is a list ...
                        for ll in np.arange(0,len(v)):
                            this_dict[index + '-' + v[ll]['valueType']] = formatValue(v[ll]['valueType'], v[ll]['value'])

                    elif isinstance(v, dict):
                        # If the field value is a dictionary ...
                        if index == 'temperature':
                            # The only field where a dictionary specifies metadata for one value
                            # e.g. the dictionary contains 'valueType', 'value' and 'valueTime'
                            # Setting the value is the easy bit ...
                            this_dict[index + '-' + v['valueType']] = formatValue(index, v['value'])
                            # But grabbing the time data is a bit more tricky because the time in %H:%M:%S is only
                            # recorded (not the day!)
                            # So, I am assuming that the date is the same date as the filename
                            try:
                                this_dict[index + '-' + v['valueType'] + '-' + 'time'] = \
                                    [dt.datetime.strptime(thisdate.strftime('%Y-%m-%d ') +
                                                         formatValue('time', v['valueTime'])[0], '%Y-%m-%d %H:%M:%S')]
                            except:
                                this_dict[index + '-' + v['valueType'] + '-' + 'time'] = \
                                    formatValue('time', v['valueTime'])
                        else:
                            # Otherwise, loop through all the keys in the dictionary
                            for k in v.keys():
                                fieldname = index + '-' + k if not k == 'value' else index
                                this_dict[fieldname] = formatValue(k, v[k])
                    else:
                        # It shouldn't arrive at this option, but if it's neither a list or a dictionary, have a guess
                        this_dict[index] = formatValue(index, v)

            #  This is checking to see if we're on the first file, if so, create a dataframe to output,
            #  otherwise append to existing
            if isinstance(outdf, dict):
                outdf = pd.DataFrame(this_dict)
            else:
                outdf = outdf.append(pd.DataFrame(this_dict), sort=False)

        # Move onto the next datetime
        # Note that the frequency of observations is set in the location_config.py file
        thisdate = thisdate + dt.timedelta(hours=freq)

    # TODO: Once we have the final output, let's add in a column for local times too

    # Return a pandas dataframe of the requested data
    return outdf


def plotStationData(df, plotsettings):
    
    '''
    Plot a time series for all the columns in the dataframe
    '''

    if not os.path.isdir(plotsettings['plotdir']):
        os.makedirs(plotsettings['plotdir'])

    ofile = plotsettings['plotdir'] + 'synops_' + \
            plotsettings['start'].strftime('%Y%m%dT%H%MZ') + '-' + \
            plotsettings['end'].strftime('%Y%m%dT%H%MZ') + '_' + str(plotsettings['station_id']) + '.png'

    title = 'Synoptic observations at station '+str(plotsettings['station_id'])
    xlab = 'Time'

    matplotlib.rcParams.update({'font.size': 7})
    plt.figure(figsize=(12,17))
    df.plot.line(x='dateTimeUTC',
                    y=['wetbulb', 'drybulb', 'pressure-mslp', 'wind-speed', 'wind-direction', 'wind-gust',
                       'rainfall-observed', 'cloud-oktas'], title=title, subplots=True)
    plt.xlabel(xlab)
    plt.savefig(ofile, dpi=400)



    # fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    #
    # sns.lineplot(x=outdf['dateTimeUTC'], y=outdf['rainfall'], data=outdf, ax=ax1)
    # sns.lineplot(x=outdf['dateTimeUTC'], y=outdf['pressure'], data=outdf, ax=ax2)
    # sns.lineplot(x=outdf['dateTimeUTC'], y=outdf['drybulb'], data=outdf, ax=ax3)

    #
    #sns.relplot(x=outdf['dateTimeUTC'], y=outdf['rainfall'], data=outdf, ax=ax1)
    #sns.relplot(x=outdf['dateTimeUTC'], y=outdf['pressure'], data=outdf, ax=ax2)
    #sns.relplot(x=outdf['dateTimeUTC'], y=outdf['drybulb'], data=outdf, ax=ax3)

# def bokeh_plot(df, settings):
#     # Possible bokeh implementation ...
#     output_file = ofile.replace('png', 'html'), title=title)
#     TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
#     fig = figure(title=title, y_axis_type="linear", plot_height=1200,
#                tools=TOOLS, plot_width=1700)
#     fig.xaxis.axis_label = 'Time'
#     fig.line()

def main(organisation, start_dt, end_dt, station_id):
    '''
    This controls all the functions that we have written. It should be generic (i.e. works for all NMSs)
    :param start_dt: datetime object specifying the start of the period
    :param end_dt: datetime object specifying the end of the period
    :param station_id: integer relating to a station (we may add in a geographical search later)
    :return: Lots of plots in a directory and an html page allowing navigation of the plots

    TODO: Geographical search of station data. Requires a file that relates station_id to lat/lons
    TODO: Retrieve UM / WRF model data and plot on the same figure (possibly a reduced set of variables)
    TODO: Create a web page of plots to allow easy viewing
    '''

    # Set some location-specific defaults
    settings = load_location_settings(organisation)

    # Get the obs data
    for st_id in station_id:

        df = getData(start_dt, end_dt, settings, st_id)

        # Get the model data

        # Make the obs only plots
        plotsettings = {'plotdir': settings['synop_path'], 'station_id': st_id, 'start': start_dt, 'end': end_dt}
        plotStationData(df, plotsettings)

        # Make the model vs obs plots


if __name__ == '__main__':

    # organisation, start_dt, end_dt, station_id
    try:
        organisation = sys.argv[1]
    except:
        # TODO: Add a function here to determine country / organisation by IP address
        # For the time being though, PAGASA data is tested with this
        organisation = 'Andy-MacBook'

    now = dt.datetime.utcnow()
    try:
        start_dt = dt.datetime.strptime(sys.argv[2], '%Y%m%d%H%M')
    except:
        start_dt = dt.datetime(2019, 11, 3, 0)
        # start_dt = now - dt.timedelta(days=7)

    try:
        end_dt = dt.datetime.strptime(sys.argv[3], '%Y%m%d%H%M')
    except:
        # end_dt = now
        end_dt = dt.datetime(2019, 11, 6, 0)

    # Allows the user to plot multiple station IDs at once
    try:
        station_id = sys.argv[4]
        if ',' in station_id:
            station_id = [int(x) for x in station_id.split(',')]
        else:
            station_id = [station_id]
    except:
        # TODO: Write a function that returns some common station IDs (e.g. Metro Manila, Cebu, etc)
        station_id = [98851, 98222]

    main(organisation, start_dt, end_dt, station_id)
