#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:36:54 2019

@author: hmiguel
"""
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import datetime as dt
import pdb
import seaborn as sns

def loadLocations(site):
    '''
    This loads data depending on the NMS that we're in
    '''
    if site == 'PAGASA':
        path = '/home/hmiguel/WCSSP/synop/'
        file_wildcard = '*.json'
        
    return [path, file_wildcard]
    
def formatValue(k, v):
    
    if k == 'rainfall' and v == '':
        v = np.NaN
    elif k == 'rainfall' and v == 'T':
        v = 0
    else:                
        v = float(v)
    
    
    if df_subset['pressure'].iloc[0] == None:
        mslp = np.NaN
    else:
        mslp = float(df_subset['pressure'].iloc[0]['mslp'])

        
    if df_subset['drybulb'].iloc[0] == None:
        drybulb = np.NaN
    else: 
        drybulb = float(df_subset['drybulb'].iloc[0]['value'])

        
    if df_subset['wetbulb'].iloc[0] == None:
        wetbulb = np.NaN
    else: 
        wetbulb = float(df_subset['wetbulb'].iloc[0]['value'])

    if df_subset['wind'].iloc[0]['speed'] == None:
        wind_speed = np.NaN
    else:
        wind_speed = float(df_subset['wind'].iloc[0]['speed'])

 
    date = [dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in df_subset['dateTimeUTC']]


    return [v]
    
def getData(start_dt, end_dt, station_id=None):
    '''
    start_dt and end_dt are datetime objects
    station_id is optional, but if given, selects just one station
    '''
    
    # Get the data
    path, file_wildcard = loadLocations('PAGASA')
    
    thisdate = start_dt
    filelist = []
    outdf = []

    while thisdate <= end_dt:
        
        thisfile, = glob.glob(path + '*' + thisdate.strftime('%Y%m%d%H%M') + file_wildcard)
        filelist.append(thisfile)
        
        df = pd.DataFrame(pd.read_json(thisfile))
        df_subset = df.loc[df['stationNumber'] == station_id]
        if df_subset.empty == False:
            print('not empty')
            
            this_dict = {}
            for index, val in df_subset.iteritems():
                #print(index)
                for v in val:
                    try:
                        for k in v.keys():
                            print(k, ': ', formatValue(k, v[k]))
                            this_dict[k] = formatValue(k, v[k])
                    except:
                        print(index, ': ', formatValue(k, v))
                        this_dict[k] = formatValue(k, v)

            if isinstance(outdf, list):               
                outdf = pd.DataFrame(this_dict)
            else:
                tmp_df = pd.DataFrame(this_dict)
                outdf = outdf.append(tmp_df)
                
            
        thisdate = thisdate + dt.timedelta(hours=3)
    
    
    # Return a pandas dataframe of the requested data
    return outdf


def plotStationData(indf):
    
    '''
    Plot a time series for all the columns in the dataframe
    '''


fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)

sns.lineplot(x=outdf['dateTimeUTC'], y=outdf['rainfall'], data=outdf, ax=ax1)
sns.lineplot(x=outdf['dateTimeUTC'], y=outdf['pressure'], data=outdf, ax=ax2)
sns.lineplot(x=outdf['dateTimeUTC'], y=outdf['drybulb'], data=outdf, ax=ax3)

#
#sns.relplot(x=outdf['dateTimeUTC'], y=outdf['rainfall'], data=outdf, ax=ax1)
#sns.relplot(x=outdf['dateTimeUTC'], y=outdf['pressure'], data=outdf, ax=ax2)
#sns.relplot(x=outdf['dateTimeUTC'], y=outdf['drybulb'], data=outdf, ax=ax3)
