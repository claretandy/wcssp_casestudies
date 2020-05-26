"""
Created on Tue Nov 12 15:39:30 2019

@author: nms6
"""
import os, sys
import datetime as dt
import location_config as config
import pandas as pd
import nrt_plots_v3 as nrtplt
import downloadSoundings
import iris.plot as iplt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import tephi

import pdb

def tephi_plot(station, date, input_dict, plot_fname):
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

    if not isinstance(date, dt.datetime):
        date = dt.datetime.strptime(date, '%Y%m%d_%H%M')

    fig,ax = plt.subplots(figsize=(10,20))
    plt.axis('off')
    
    # Input tephigram modifications here
    tpg = tephi.Tephigram(figure=fig, anchor=[(1000, -10), (100, -50)])
    tephi.MIN_THETA = -10
    tephi.WET_ADIABAT_SPEC = [(5, None)]
    tephi.MAX_WET_ADIABAT = 60
    tephi.MIXING_RATIO_LINE.update({'linestyle': '--'})

    dews = [d for d in zip(input_dict['pressure'], input_dict['dew_point'])]
    temps = [t for t in zip(input_dict['pressure'], input_dict['temperature'])]
    tbarbs = [brb for brb in zip(input_dict['wind_speed'], input_dict['wind_dir'], input_dict['pressure'])]

    # pdb.set_trace()

    dprofile = tpg.plot(dews)
    tprofile = tpg.plot(temps)
    tprofile.barbs(tbarbs, color='black', linewidth=0.1)

    # for key,data in input_dict.items():
    #
    #     if style_dict is None:
    #         tpg.plot(data, label=key)
    #     else:
    #         tpg.plot(data, color=style_dict[key]['c'], linestyle=style_dict[key]['ls'], label=key)

    plt.title('Station: '+str(station.wigosStationIdentifier)+'\n'+station['name']+'\n'+station.territory, loc='left')
    # plt.title('Station: ' + str(station.wigosStationIdentifier) + ' \n ' + 'station.name', loc='left')
    plt.title('Valid time\n'+date.strftime('%H:%M (UTC)')+'\n'+date.strftime('%Y-%m-%d'), loc='right')
    plt.savefig(plot_fname, bbox_inches='tight')

    plt.close(fig)

def getData(start_dt, end_dt, event_domain, settings):
    """
    Runs functions to get data from each organisation (which might be in different formats)
    :param start_dt: datetime object
    :param end_dt: datetime object
    :param event_domain: list containing float values of [xmin, ymin, xmax, ymax]
    :param settings: settings read from the config file
    :return: pandas dataframe of all stations and all datetimes available
    """
    organisation = settings['organisation']

    # This might be necessary to get a list of upper air stations within the event domain
    stations_df = downloadSoundings.getUpperAirStations(event_domain)

    if organisation == 'BMKG':
        data = downloadSoundings.main(start_dt, end_dt, event_domain, settings)['data']
        # Replace with the following when available
        # data = getData_BMKG(start_dt, end_dt, settings, stations_df)
    elif organisation == 'PAGASA':
        data = downloadSoundings.main(start_dt, end_dt, event_domain, settings)['data']
        # Replace with the following when available
        # data = getData_PAGASA(start_dt, end_dt, settings, stations_df)
    elif organisation == 'MMD':
        data = downloadSoundings.main(start_dt, end_dt, event_domain, settings)['data']
        # Replace with the following when available
        # data = getData_MMD(start_dt, end_dt, settings, stations_df)
    elif organisation == 'UKMO':
        data = downloadSoundings.main(start_dt, end_dt, event_domain, settings)['data']
    else:
        data = downloadSoundings.main(start_dt, end_dt, event_domain, settings)['data']

    try:
        return data, stations_df
    except:
        return 'Something went wrong with retrieving the data'


def data_to_dict(df):
    """
    Translates a dataframe with different column names in to a dictionary with standard key names
    :param df: pandas dataframe with columns containing data for tephigram plots
    :return: dictionary with standardised keys
    """
    out_dict = {}
    cols = df.columns
    lut = {'pressure': ['P', 'PRES', 'PRESSURE'],
           'height': ['HGHT', 'HEIGHT'],
           'temperature': ['T', 'TEMP', 'TEMPERATURE'],
           'dew_point': ['DWPT', 'DEWPOINT', 'DEWPOINTTEMP'],
           'wind_dir': ['DRCT', 'WDIR', 'WINDDIR'],
           'wind_speed': ['SKNT', 'WSPD', 'WINDSPD']}

    for k in lut.keys():
        try:
            col, = [x for x in lut[k] if x in cols]
            out_dict.update({k : df[col].to_list()})
        except:
            print(k + ' column name not found, if it exists in df, please add to the look up table')
            if k in ['pressure', 'temperature']:
                return('Can\'t plot a tephi without ' + k)

    return out_dict

def getData_MMD(start_dt, end_dt, settings, stations_df):
    """
    Function for MMD to write in order to retrieve local sounding data
    :param start_dt:
    :param end_dt:
    :param settings:
    :param stations_df: Pandas dataframe of all the upper stations within the event_domain
    :return: A pandas dataframe containing data for all stations for all dates.
    Column names need to match those output by downloadSoundings.main()
    """

def getData_PAGASA(start_dt, end_dt, settings, stations_df):
    """
    Function for PAGASA to write in order to retrieve local sounding data
    :param start_dt:
    :param end_dt:
    :param settings:
    :param stations_df: Pandas dataframe of all the upper stations within the event_domain
    :return: A pandas dataframe containing data for all stations for all dates.
    Column names need to match those output by downloadSoundings.main()
    """

def getData_BMKG(start_dt, end_dt, settings, stations_df):
    """
    Function for BMKG to edit in order to retrieve local sounding data
    :param start_dt:
    :param end_dt:
    :param settings:
    :param stations_df: Pandas dataframe of all the upper stations within the event_domain
    :return: A pandas dataframe containing data for all stations for all dates.
    Column names need to match those output by downloadSoundings.main()
    """

    # TODO: Set up automatic download of data files to the Data directory. Might be a separate function
    
    # For testing though, use the following sample file ...
    # e.g.
    # function_to_send_url_request(start_dt, end_dt, station_id)
    infile = 'SampleData/upper-air/BMKG/sample_upper_revised_bmkg.csv'
    df = pd.read_csv(infile)
    column_names = ['ID', 'StationNumber', 'StationName', 'Latitude', 'Longitude', 'Date',
                    'Pressure', 'temp', 'dewpt_temp', 'wind_dir', 'wind_speed']
    df.columns = column_names

    if stn_id:
        df_subset = df.loc[df['StationNumber'] == stn_id]
    else:
        df_subset = df

    # TODO: Format the date field and subset according to the date range given


    # Assume that we have one date?
    # dates = list(set(df_subset['Date']))

    #  TODO: Once we have the final output, let's add in a column for local times too

    # Create an output dictionary
    out_dict = {
        'datetime': [dt.datetime.strptime(x, '%Y%m%d_%H%M') for x in df_subset['Date'].to_list()],
        'pressure': df_subset['Pressure'].to_list(),
        'temperature': df_subset['temp'].to_list(),
        'dew_point': df_subset['dewpt_temp'].to_list(),
        'wind_dir': df_subset['wind_dir'].to_list(),
        'wind_speed': df_subset['wind_speed'].to_list()
    }

    # Return a dictionary of the requested data
    return out_dict

def plot_station_map(stations, event_domain, map_plot_fname):
    """
    Plots the locations of upper air stations within the event domain
    :param stations: pandas dataframe of upper air stations within event domain
    :param map_plot_fname: filename for plot output
    :return: File name of resulting plot
    """

    filedir = os.path.dirname(map_plot_fname)
    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    # Put domain into correct format for plotting
    domain = [event_domain[0], event_domain[2], event_domain[1], event_domain[3]]

    # Now do the plotting
    fig = plt.figure(figsize=nrtplt.getFigSize(event_domain), dpi=96)
    pltax = plt.axes(projection=ccrs.PlateCarree())

    for i, row in stations.iterrows():
        pltax.plot(row.longitude, row.latitude, marker='o', color='blue', transform=ccrs.PlateCarree())
        pltax.text(row.longitude + 0.2, row.latitude, row['name'], transform=ccrs.PlateCarree())

    plt.title('Upper Air Stations')
    plt.xlabel('longitude / degrees')
    plt.ylabel('latitude / degrees')
    ax = plt.gca()

    ax.set_extent(domain)
    borderlines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')
    ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
    ax.coastlines(resolution='50m', color='black')
    gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    fig.savefig(map_plot_fname, bbox_inches='tight')
    plt.close(fig)


def main(start_dt, end_dt, event_domain, event_name, organisation):

    # For testing
    # start_dt = dt.datetime(2020, 5, 19, 0)
    # end_dt = dt.datetime(2020, 5, 20, 0)
    # event_name = 'PeninsulaMalaysia/20200519_test'
    # event_domain = [100, 0, 110, 10]
    # organisation = 'UKMO'

    # Set some location-specific defaults
    settings = config.load_location_settings(organisation)

    # Get Data
    data, stations = getData(start_dt, end_dt, event_domain, settings)
    # Plot map of stations
    map_plot_fname = settings['plot_dir'] + event_name + '/upper-air/station_map.png'
    plot_station_map(stations, event_domain, map_plot_fname)

    # Loop through station ID(s)
    for i, station in stations.iterrows():
        # Get the obs data
        print(station)
        stn_id = station['wigosStationIdentifier']
        datesnp = data[data.station_id == str(stn_id)].datetimeUTC.unique()
        dates = pd.DatetimeIndex(datesnp).to_pydatetime() # Converts numpy.datetime64 to datetime.datetime
        for thisdt in dates:
            print(thisdt)
            mysubset = data[(data.station_id == str(stn_id)) & (data.datetimeUTC == thisdt)]
            dict_to_plot = data_to_dict(mysubset)
            plot_fname = settings['plot_dir'] + event_name + '/upper-air/' + thisdt.strftime('%Y%m%dT%H%MZ') + '_' + str(stn_id) + '.png'
            tephi_plot(station, thisdt, dict_to_plot, plot_fname)


if __name__ == '__main__':

    now = dt.datetime.utcnow()
    try:
        start_dt = dt.datetime.strptime(sys.argv[1], '%Y%m%d%H%M')
    except:
        # For testing
        start_dt = dt.datetime(2019, 11, 3, 0)

    try:
        end_dt = dt.datetime.strptime(sys.argv[2], '%Y%m%d%H%M')
    except:
        # For testing
        end_dt = dt.datetime(2019, 11, 6, 0)

    try:
        domain_str = sys.argv[3]
        event_domain = [float(x) for x in domain_str.split(',')]
    except:
        # For testing
        event_domain = [100, 0, 110, 10]

    try:
        event_name = sys.argv[4]
    except:
        # For testing
        event_name = 'PeninsulaMalaysia/20190122_Johor'

    try:
        organisation = sys.argv[5]
    except:
        # TODO: Add a function here to determine country / organisation by IP address
        #  For the time being though, Wyoming data is tested with this
        organisation = 'UKMO'

    main(start_dt, end_dt, event_domain, event_name, organisation)
