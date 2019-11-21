"""
Created on Tue Nov 12 15:39:30 2019

@author: nms6
"""
import sys
sys.path.append('./tephi_module')
from tephi_module import tephi
import datetime as dt
import matplotlib.pyplot as plt
import location_config as config
import pandas as pd
import glob

def tephi_plot(station, date, input_dict, plot_fname, style_dict=None):
    """
    Plot T-ϕ-gram using modified tephi scripts.
    
    station:    str, station name
    date:       date str w/ format %Y%m%d_%H%M
    input_dict:
        keys:   data type
        values: 2D numpy array of [pressure level, temperature] 
    style_dict:
        keys:   data type
        values: dictionary {'c':, 'ls':}
    """

    date = dt.datetime.strptime(date, '%Y%m%d_%H%M')
    
    fig,ax = plt.subplots(figsize=(10,20))
    plt.axis('off')
    
    # Input tephigram modifications here
    tpg = tephi.Tephigram(figure=fig, anchor=[(1000, -10), (100, -50)])
    tephi.MIN_THETA = -10
    tephi.WET_ADIABAT_SPEC = [(5, None)]
    tephi.MAX_WET_ADIABAT = 60
    tephi.MIXING_RATIO_LINE.update({'linestyle': '--'})

    for key,data in input_dict.items():
        
        if style_dict is None:
            tpg.plot(data, label=key)
        else:
            tpg.plot(data, color=style_dict[key]['c'], linestyle=style_dict[key]['ls'], label=key)
    
    plt.title('Station: '+station, loc='left')
    plt.title('Valid date: '+date.strftime('%Y%m%d %HZ'), loc='right')
    plt.savefig(plot_fname, bbox_inches='tight')


def getData(start_dt, end_dt, settings, station_id):
    organisation = settings['organisation']
    if organisation == 'BMKG':
        data = getData_BMKG(start_dt, end_dt, settings, station_id)
    elif organisation == 'PAGASA':
        data = getData_PAGASA(start_dt, end_dt, settings, station_id)
    elif organisation == 'MMD':
        data = getData_MMD(start_dt, end_dt, settings, station_id)
    elif organisation == 'UKMO':
        data = getData_UKMO(start_dt, end_dt, settings, station_id)
    else:
        print("Can\'t find the function to read the data")

    try:
        return data
    except:
        return

def getData_BMKG(start_dt, end_dt, settings, st_id):

    '''
    This script connects to the internal FTP or file system and returns a dictionary of sounding data

    start_dt and end_dt are datetime objects
    station_id is optional, but if given, selects just one station
    '''

    # TODO: Set up automatic download of data files to the Data directory. Might be a separate function
    
    # For testing though, use the following sample file ...
    # e.g.
    # function_to_send_url_request(start_dt, end_dt, station_id)
    infile = 'SampleData/upper-air/sample_upper_revised_bmkg.csv'
    df = pd.read_csv(infile)
    column_names = ['ID', 'StationNumber', 'StationName', 'Latitude', 'Longitude', 'Date',
                    'Pressure', 'temp', 'dewpt_temp', 'wind_dir', 'wind_speed']
    df.columns = column_names

    if st_id:
        df_subset = df.loc[df['StationNumber'] == st_id]
    else:
        df_subset = df

    # Assume that we have one date?
    dates = list(set(df_subset['Date']))

    #  TODO: Once we have the final output, let's add in a column for local times too

    # Create an output dictionary
    out_dict = {
        'pressure': df_subset['Pressure'].to_list(),
        'temperature': df_subset['temp'].to_list(),
        'dew_point': df_subset['dewpt_temp'].to_list(),
        'wind_dir': df_subset['wind_dir'].to_list(),
        'wind_speed': df_subset['wind_speed'].to_list()
    }
    # Return a pandas dataframe of the requested data
    return out_dict, dates

def main(organisation, start_dt, end_dt, station_id):
    # Set some location-specific defaults
    settings = config.load_location_settings(organisation)

    # Get the obs data
    for st_id in station_id:
        input_dict, dates = getData(start_dt, end_dt, settings, st_id)
        for thisdt in dates:
            plot_fname = settings['plot_dir'] + '/upper-air/' + thisdt + '_' + str(st_id) + '.png'
            tephi_plot(st_id, thisdt, input_dict, plot_fname, style_dict=None)
######################################################################################
#def tephi_plot(station, date, barbs, TEPHI, style_dict=None):

if __name__ == '__main__':
    # organisation, start_dt, end_dt, station_id
    try:
        organisation = sys.argv[1]
    except:
        # TODO: Add a function here to determine country / organisation by IP address
        #  For the time being though, PAGASA data is tested with this
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

    #  Allows the user to plot multiple station IDs at once
    try:
        station_id = sys.argv[4]
        if ',' in station_id:
            station_id = [int(x) for x in station_id.split(',')]
        else:
            station_id = [station_id]
    except:
        #  TODO: Write a function that returns some common station IDs (e.g. Cengkareng, Hasanuddin, etc)
        station_id = [96749, 97180]

    main(organisation, start_dt, end_dt, station_id)


