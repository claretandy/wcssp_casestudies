"""
Created on Tue Nov 12 15:39:30 2019

@author: nms6
"""
import os, sys
import datetime as dt

import load_data
import location_config as config
import iris
import pandas as pd
import plot_precip
import downloadSoundings
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import tephi
import std_functions as sf
import run_html as html
import pdb

def tephi_plot(station, date, input, plot_fname):
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

    fig,ax = plt.subplots(figsize=(10,15), dpi=100)
    plt.axis('off')
    
    # Input tephigram modifications here
    tpg = tephi.Tephigram(figure=fig, anchor=[(1000, -10), (100, -50)])
    tephi.MIN_THETA = -10
    tephi.WET_ADIABAT_SPEC = [(5, None)]
    tephi.MAX_WET_ADIABAT = 60
    tephi.MIXING_RATIO_LINE.update({'linestyle': '--'})

    # Make sure that we are only plotting rows with all records
    input = input.loc[input.PRES.notnull() & input.DWPT.notnull() & input.TEMP.notnull() & input.SKNT.notnull() & input.DRCT.notnull()]

    # Get a unique list of model ids
    model_ids = pd.unique(input.model_id.loc[(input.model_id != 'observation') & (input.model_id != 'analysis')])

    # Plot each model, with different opacity if multiple lead times available
    colours = plt.get_cmap('tab10')
    for i, m in enumerate(model_ids):
        mod = input.loc[input.model_id == m]
        fclts = pd.unique(mod.fcast_lead_time)
        col = colours.colors[i]
        alphas = np.linspace(1.0, 0.3, len(fclts)) if len(fclts) > 3 else np.linspace(1.0, 0.5, len(fclts))
        # Need to put something here to fix colours and opacity
        for a, fclt in enumerate(fclts):
            modfc = mod.loc[mod.fcast_lead_time == fclt]
            if not modfc.empty:
                dews = [d for d in zip(modfc['PRES'], modfc['DWPT'])]
                temps = [t for t in zip(modfc['PRES'], modfc['TEMP'])]
                tbarbs = [brb for brb in zip(modfc['SKNT'], modfc['DRCT'], modfc['PRES'])]
                tpg.plot(dews, linestyle='--', color=(col[0], col[1], col[2], alphas[a]))
                tprofile = tpg.plot(temps, color=(col[0], col[1], col[2], alphas[a]), label=m + ' (T+' + str(fclt) + ')')
                tprofile.barbs(tbarbs, color=(col[0], col[1], col[2], alphas[a]), linewidth=0.5)

    # Next, plot the analysis if it exists
    ana = input.loc[input.model_id == 'analysis']
    if not ana.empty:
        dews = [d for d in zip(ana['PRES'], ana['DWPT'])]
        temps = [t for t in zip(ana['PRES'], ana['TEMP'])]
        tbarbs = [brb for brb in zip(ana['SKNT'], ana['DRCT'], ana['PRES'])]
        tpg.plot(dews, color='purple', linestyle='--')
        tprofile = tpg.plot(temps, color='purple', label='UM Analysis')
        tprofile.barbs(tbarbs, color='purple', linewidth=0.5)

    # Finally, plot the observations if they exist
    obs = input.loc[input.model_id == 'observation']
    if not obs.empty:
        dews = [d for d in zip(obs['PRES'], obs['DWPT'])]
        temps = [t for t in zip(obs['PRES'], obs['TEMP'])]
        tbarbs = [brb for brb in zip(obs['SKNT'], obs['DRCT'], obs['PRES'])]
        tpg.plot(dews, color='black', linestyle='--')
        tprofile = tpg.plot(temps, color='black', label='Radiosonde')
        tprofile.barbs(tbarbs, color='black', linewidth=0.5)

    plt.title('Station: '+str(station.wigosStationIdentifier)+'\n'+station['name']+'\n'+station.territory, loc='left')
    # plt.title('Station: ' + str(station.wigosStationIdentifier) + ' \n ' + 'station.name', loc='left')
    plt.title('Valid time\n'+date.strftime('%H:%M (UTC)')+'\n'+date.strftime('%Y-%m-%d'), loc='right')
    plt.savefig(plot_fname, bbox_inches='tight')

    plt.close(fig)


def getModelData(start_dt, end_dt, event_name, bbox, locations, settings, model_id='all'):
    """
    Looks in the data directory for UM model data, and returns
    vertical profiles for the time period and locations specified
    :param start_dt: datetime
    :param end_dt: datetime
    :param event_name: string name of the events as '<region>/<datetime>_<location>'
    :param bbox: dictionary specifying bounding box that we want to plot
            e.g. {'xmin': 99, 'ymin': 0.5, 'xmax': 106, 'ymax': 7.5}
    :param locations: either a dataframe of stations or a list of lat/lon tuples e.g. [(x1,y1),(x2,y2)]
    :param settings: settings read from the config file
    :param model_id: Can be any from the following ['analysis', 'ga6', 'ga7', 'km4p4', 'km1p5', 'all']
    :return: pandas dataframe of all available model data
    """

    # Timestep defines how frequently we want to return UM data (starting at 00Z)
    # Chosen 6 hourly here to match the analysis
    timestep = 6

    # Set the variables (note that the analysis doesn't have specific humidity)
    vars = ['rh-wrt-water', 'rh-wrt-ice', 'specific-humidity-levels', 'temp-levels', 'Uwind-levels', 'Vwind-levels']
    # vars = ['rh-wrt', 'temp-levels', 'Uwind-levels', 'Vwind-levels'] if model_id == 'analysis' else vars

    # Load the model cubes into a dictionary
    print('Loading model data ...')
    alldata_allmodels = load_data.unified_model(start_dt, end_dt, event_name, settings, bbox=bbox, model_id=model_id, var=vars, timeclip=False, aggregate=False)

    # Sets up an empty dataframe to put the data in
    column_names = ['stn_id', 'model_id', 'valid_datetimeUTC', 'fcast_lead_time', 'PRES', 'value', 'variable']
    df = pd.DataFrame(columns=column_names)

    print('Extracting model data for upper air stations ...')
    for m in alldata_allmodels.keys():

        alldata = alldata_allmodels[m]
        if m == 'analysis':
            myu, = list(set([alldata[k].coord('forecast_reference_time').units for k in alldata.keys()]))
        else:
            # This covers model data that has multiple initialisations
            myu, = list(set([cube.coord('forecast_reference_time').units for k in alldata.keys() for cube in alldata[k]]))
        # fcrt = np.unique(np.array([cube.coord('forecast_reference_time').points for k in alldata.keys() for cube in alldata[k]]).flatten())

        # Sets the start adjusted to every <timestep> hours
        validtimes = sf.make_timeseries(start_dt, end_dt, timestep)

        for vt in validtimes:

            tcon = iris.Constraint(time=lambda cell: cell.point == vt)

            # Create an empty dataframe for this model / validtime
            # Save to a unique file

            # Loop through each location
            for i, row in locations.iterrows():
                print('   Extracting:', m, 'for', row['name'])
                sample_points = [('latitude', row['latitude']), ('longitude', row['longitude'])]

                for k in alldata.keys():
                    print('   ... extracting:', row['name'], m, vt, k)
                    cubelist = alldata[k]
                    try:
                        cubes = [cube for cube in cubelist if myu.date2num(vt) in cube.coord('time').points]
                    except TypeError:
                        # NB: For analysis data, a merged cube is output (not a cubelist)
                        cubes = [cubelist]
                    except:
                        pdb.set_trace()
                        continue

                    for cube in cubes:
                        # Loops through the different forecast reference times for this variable and model
                        if cube:
                            cube = cube.extract(tcon)
                            try:
                                cube = cube.interpolate(sample_points, iris.analysis.Linear())
                            except:
                                continue
                            dfk_cube = pd.DataFrame({'stn_id': row['wigosStationIdentifier'],
                                                'model_id': m,
                                                'valid_datetimeUTC': vt,
                                                'fcast_lead_time': cube.coord('forecast_period').points[0],
                                                'PRES': cube.coord('pressure').points,
                                                'value': cube.data.data,
                                                'variable': k })
                            df = pd.concat([df, dfk_cube])

    # Change the pandas DataFrame to wide format (i.e. variables are now columns)
    df_pivot = df.pivot_table(index=['stn_id', 'model_id', 'valid_datetimeUTC', 'fcast_lead_time', 'PRES'], columns='variable', values='value')

    # Calculate RH and Td if missing
    # This gets RH from Q, T and P
    if not 'specific-humidity-levels' in df_pivot.columns:
        df_pivot['specific-humidity-levels'] = np.nan

    notna = df_pivot['specific-humidity-levels'].notnull() & df_pivot['temp-levels'].notnull() & df_pivot['rh-wrt-water-levels'].isnull()
    q = df_pivot[notna]['specific-humidity-levels'].to_numpy()
    t = df_pivot[notna]['temp-levels'].to_numpy()
    p = df_pivot[notna].index.get_level_values('PRES').to_numpy()

    if not df_pivot[notna].empty:
        df_pivot.loc[notna, 'RELH'] = sf.compute_rh(q, t, p) # es_eqtn='cc1'

    # UM Analysis doesn't output specific humidity, but does have RH wrt water and ice
    hasrh = df_pivot['rh-wrt-water-levels'].notnull() | df_pivot['rh-wrt-ice-levels'].notnull()
    df_pivot.loc[hasrh, 'RELH'] = df_pivot[hasrh][['rh-wrt-water-levels', 'rh-wrt-ice-levels']].max(axis=1) # + df_pivot[notna]['rh-wrt-ice-levels']

    # Now calculate dewpoint temperature
    hasrht = df_pivot['RELH'].notnull() & df_pivot['temp-levels'].notnull()
    df_pivot.loc[hasrht, 'DWPT'] = sf.compute_td(df_pivot[hasrht]['RELH'].to_numpy(), df_pivot[hasrht]['temp-levels'].to_numpy())

    # Calculate speed and direction from u and v
    wnotna = df_pivot['Uwind-levels'].notnull() & df_pivot['Vwind-levels'].notnull()
    df_pivot.loc[wnotna, 'DRCT'] = sf.compute_wdir(df_pivot[wnotna]['Uwind-levels'].to_numpy(), df_pivot[wnotna]['Vwind-levels'].to_numpy())
    df_pivot.loc[wnotna, 'SKNT'] = sf.compute_wspd(df_pivot[wnotna]['Uwind-levels'].to_numpy(), df_pivot[wnotna]['Vwind-levels'].to_numpy(), units='knots')

    # Put all the data in the MultiIndex back into the dataframe table
    df_pivot.reset_index(inplace=True)
    df_pivot = df_pivot.rename(columns={'stn_id': 'station_id', 'valid_datetimeUTC': 'datetimeUTC', 'temp-levels': 'TEMP'})

    # Specific Humidity is the same as the mass mixing ration of water vapour to total air
    if 'specific-humidity-levels' in df_pivot.columns:
        df_pivot = df_pivot.rename(columns={'specific-humidity-levels': 'MIXR'})

    # Convert some units to allow easier comparison with observations
    # Convert Kelvin to degrees C
    df_pivot['TEMP'] = df_pivot['TEMP'] - 273.15
    # Convert Q to g/kg from kg/kg
    df_pivot['MIXR'] = df_pivot['MIXR'] * 1000.

    return df_pivot


def getObsData(start_dt, end_dt, event_domain, settings):
    """
    Runs functions to get data from each organisation (which might be in different formats)
    :param start_dt: datetime object
    :param end_dt: datetime object
    :param event_domain: list containing float values of [xmin, ymin, xmax, ymax]
    :param settings: settings read from the config file
    :return: pandas dataframe of all stations and all datetimes available
    """
    organisation = settings['organisation']

    # This might be necessary to get a list of upper air stations within the event bbox
    stations_df = downloadSoundings.getUpperAirStations(event_domain)

    if organisation == 'BMKG':
        data = downloadSoundings.main(start_dt, end_dt, event_domain, settings)['data']
        # Replace with the following when available
        # data = getObsData_BMKG(start_dt, end_dt, settings, stations_df)

    elif organisation == 'PAGASA':
        data = downloadSoundings.main(start_dt, end_dt, event_domain, settings)['data']
        # Replace with the following when available
        # data = getObsData_PAGASA(start_dt, end_dt, settings, stations_df)

    elif organisation == 'MMD':
        data = downloadSoundings.main(start_dt, end_dt, event_domain, settings)['data']
        # Replace with the following when available
        # data = getObsData_MMD(start_dt, end_dt, settings, stations_df)

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
    lut = {'pressure': ['P', 'PRES', 'PRESSURE', 'pressure'],
           'height': ['HGHT', 'HEIGHT'],
           'temperature': ['T', 'TEMP', 'TEMPERATURE', 'temp-levels'],
           'dew_point': ['DWPT', 'DEWPOINT', 'DEWPOINTTEMP', 'td'],
           'wind_dir': ['DRCT', 'WDIR', 'WINDDIR', 'wdir'],
           'wind_speed': ['SKNT', 'WSPD', 'WINDSPD', 'wspd']}

    for k in lut.keys():
        try:
            col, = [x for x in lut[k] if x in cols]
            out_dict.update({k : df[col].to_list()})
        except:
            print(k + ' column name not found, if it exists in df, please add to the look up table')
            if k in ['pressure', 'temperature']:
                return('Can\'t plot a tephi without ' + k)

    return out_dict

def getObsData_MMD(start_dt, end_dt, settings, stations_df):
    """
    Function for MMD to write in order to retrieve local sounding data
    :param start_dt:
    :param end_dt:
    :param settings:
    :param stations_df: Pandas dataframe of all the upper stations within the bbox
    :return: A pandas dataframe containing data for all stations for all dates.
    Column names need to match those output by downloadSoundings.main()
    """

def getObsData_PAGASA(start_dt, end_dt, settings, stations_df):
    """
    Function for PAGASA to write in order to retrieve local sounding data
    :param start_dt:
    :param end_dt:
    :param settings:
    :param stations_df: Pandas dataframe of all the upper stations within the bbox
    :return: A pandas dataframe containing data for all stations for all dates.
    Column names need to match those output by downloadSoundings.main()
    """

def getObsData_BMKG(start_dt, end_dt, settings, stations_df):
    """
    Function for BMKG to edit in order to retrieve local sounding data
    :param start_dt:
    :param end_dt:
    :param settings:
    :param stations_df: Pandas dataframe of all the upper stations within the bbox
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
    Plots the locations of upper air stations within the event bbox
    :param stations: pandas dataframe of upper air stations within event bbox
    :param map_plot_fname: filename for plot output
    :return: File name of resulting plot
    """

    filedir = os.path.dirname(map_plot_fname)
    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    # Put bbox into correct format for plotting
    domain = [event_domain[0], event_domain[2], event_domain[1], event_domain[3]]

    # Now do the plotting
    fig = plt.figure(figsize=plot_precip.getFigSize(event_domain), dpi=96)
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
    gl.top_labels = False
    gl.left_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    fig.savefig(map_plot_fname, bbox_inches='tight')
    plt.close(fig)


def merge_data_for_plotting(obsdata, modeldata):

    # Add a column to the obsdata for fcast lead time
    obsdata['fcast_lead_time'] = np.nan
    obsdata['model_id'] = 'observation'
    result = pd.concat([obsdata, modeldata], ignore_index=True, sort=True)

    cols = ['station_id', 'datetimeUTC', 'fcast_lead_time', 'model_id', 'PRES', 'DWPT', 'TEMP', 'DRCT', 'SKNT', 'RELH', 'MIXR']
    cols = [c for c in cols if c in result.columns]
    cols.extend([c for c in result.columns if not c in cols])
    result = result[cols]

    return result

def main(start_dt, end_dt, event_domain, event_name, organisation):

    # For testing
    # start_dt = dt.datetime(2020, 5, 19, 0)
    # end_dt = dt.datetime(2020, 5, 20, 0)
    # event_name = 'PeninsulaMalaysia/20200520_Johor'
    # bbox = [99, 0.5, 106, 7.5]
    # organisation = 'UKMO'

    # Set some location-specific defaults
    settings = config.load_location_settings(organisation)

    # Set model ids to plot
    model_ids = ['analysis', 'ga7', 'km4p4']

    # Get Data
    obsdata, stations = getObsData(start_dt, end_dt, event_domain, settings)
    modeldata = getModelData(start_dt, end_dt, event_name, event_domain, stations, settings, model_id=model_ids)
    data2plot = merge_data_for_plotting(obsdata, modeldata)

    # Plot map of stations
    map_plot_fname = settings['plot_dir'] + event_name + '/upper-air/station_map.png'
    plot_station_map(stations, event_domain, map_plot_fname)

    # Output list of filenames for html page
    ofiles = []

    # Loop through station ID(s)
    for i, station in stations.iterrows():

        # Get the obs data
        print(station['name'] + ', ' + station['territory'])
        stn_id = station['wigosStationIdentifier']

        # Get a list of unique datetimes
        datesnp = pd.unique(data2plot['datetimeUTC'])
        dates = sorted(pd.DatetimeIndex(datesnp).to_pydatetime())  # Converts numpy.datetime64 to datetime.datetime

        for thisdt in dates:

            print(thisdt)
            this_dt_fmt = thisdt.strftime('%Y%m%dT%H%MZ')

            # Create a boolean list of records for this station and datetime
            stndt = (data2plot.station_id == str(stn_id)) & (data2plot.datetimeUTC == thisdt)

            # Plot just the observation
            asubset = data2plot.loc[stndt & (data2plot.model_id == 'observation')]
            if not asubset.empty:
                plot_fname = sf.make_outputplot_filename(event_name, this_dt_fmt, 'Radiosonde',
                                             station['name'], 'Instantaneous', 'upper-air', 'tephigram', 'T+0')
                if not os.path.isfile(plot_fname):
                    tephi_plot(station, thisdt, asubset, plot_fname)
                ofiles.append(plot_fname)

            # Plot observation + analysis
            obsana = ((data2plot.model_id == 'observation') | (data2plot.model_id == 'analysis'))
            asubset = data2plot.loc[stndt & obsana]
            if not asubset.empty:
                plot_fname = sf.make_outputplot_filename(event_name, this_dt_fmt, 'Radiosonde+Analysis',
                                             station['name'], 'Instantaneous', 'upper-air', 'tephigram', 'T+0')
                if not os.path.isfile(plot_fname):
                    tephi_plot(station, thisdt, asubset, plot_fname)
                ofiles.append(plot_fname)

            # Plot observation + analysis + models @ multiple lead times (T+0-24, 24-48, 48-72, 72-96, 96-120)
            fclts = np.arange(0, 120, 24)
            for fclt_start in fclts:
                fclt_end = fclt_start + 24
                fc = (data2plot.fcast_lead_time > fclt_start) & (data2plot.fcast_lead_time <= fclt_end) & (data2plot.model_id != 'analysis')
                asubset = data2plot.loc[stndt & (fc | obsana)]
                if not asubset.empty:
                    plot_fname = sf.make_outputplot_filename(event_name, this_dt_fmt, 'All-Models',
                                             station['name'], 'Instantaneous', 'upper-air', 'tephigram', 'T+'+str(fclt_end))
                    if not os.path.isfile(plot_fname):
                        tephi_plot(station, thisdt, asubset, plot_fname)

                    ofiles.append(plot_fname)

            # Plot observation + analysis + models @ all lead times
            asubset = data2plot.loc[stndt]
            if not asubset.empty:
                plot_fname = sf.make_outputplot_filename(event_name, this_dt_fmt, 'All-Models',
                                             station['name'], 'Instantaneous', 'upper-air', 'tephigram', 'All-FCLT')
                if not os.path.isfile(plot_fname):
                    tephi_plot(station, thisdt, asubset, plot_fname)

                ofiles.append(plot_fname)

    html.create(ofiles)

if __name__ == '__main__':

    now = dt.datetime.utcnow()
    try:
        start_dt = dt.datetime.strptime(sys.argv[1], '%Y%m%d%H%M')
    except:
        # For testing
        start_dt = now - dt.timedelta(days=1)

    try:
        end_dt = dt.datetime.strptime(sys.argv[2], '%Y%m%d%H%M')
    except:
        # For testing
        end_dt = now

    try:
        domain_str = sys.argv[3]
        event_domain = [float(x) for x in domain_str.split(',')]
    except:
        # For testing
        event_domain = [90, -10, 120, 10]

    try:
        event_name = sys.argv[4]
    except:
        # For testing
        event_name = 'monitoring/realtime'

    try:
        organisation = sys.argv[5]
    except:
        # TODO: Add a function here to determine country / organisation by IP address
        #  For the time being though, Wyoming data is tested with this
        organisation = 'UKMO'

    main(start_dt, end_dt, event_domain, event_name, organisation)
