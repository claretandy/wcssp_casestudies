import os, sys
import datetime as dt
import urllib.request
import location_config as config
from pathlib import Path
import pandas as pd
import json
import std_functions as sf
import numpy as np
import pdb

def getFields(klist, data):
    """
    Helper function for getUpperAirStations. Subsets the full dictionary returned by the OSCAR API according to a list of field names, and returns a pandas dataframe.
    :param klist: lsit of fields to subset from the data
    :param data: dictionary containing json data returned by the OSCAR API
    :return: pandas dataframe of upper air station metadata
    """

    df = pd.DataFrame(columns=klist)
    # Loop through all the items in the dictionary
    for x in data:
        line = {}
        # Loop through all the items in the list
        for k in klist:
            try:
                val = x[k]
            except:
                val = 'NA'
            # There is potentially more than one identifier, so this takes the one tagged as 'primary'.
            # If that tag doesn't exist, it will just take the first element.
            # Also, note that the wigosStationIdentifier is quite a long number, but Wyoming only uses the 3rd element
            # of it
            if k == 'wigosStationIdentifier' and isinstance(x[k+'s'], list):
                for item in x[k+'s']:
                    if item['primary']:
                        val = item['wigosStationIdentifier'].split('-')[3]
                    else:
                        val = x[k][0]['wigosStationIdentifier'].split('-')[3]
            line[k] = val

        df = df.append(line, ignore_index=True)

    # Make sure the following fields are recognised as numeric ...
    for k in ['elevation', 'latitude', 'longitude']:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors='coerce')

    return df


def getUpperAirStations(event_domain):
    """
    Uses WMO OSCAR API to get the station IDs for upper air stations within our event bbox
    :param event_domain: list of float or int values ordered as follows: [xmin, ymin, xmax, ymax]
    :param odir: output directory in the local file system
    :return: pandas dataframe containing the data
    """

    xmin, ymin, xmax, ymax = event_domain
    url='https://oscar.wmo.int/surface/rest/api/search/station?latitudeMin='+str(ymin)+\
        '&longitudeMin='+str(xmin)+'&latitudeMax='+str(ymax)+'&longitudeMax='+str(xmax)+\
        '&stationClass=upperAirRadiosonde'

    try:
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError:
        print('Unable to download due to an HTTP error')
        return None
    except:
        print('Unknown error occurred while downloading')
        return None

    data = json.load(response)

    klist = ['name', 'territory', 'latitude', 'longitude', 'elevation', 'declaredStatus', 'wigosStationIdentifier']
    df = getFields(klist, data)

    # Check WMO API returned the correct stations
    # (seems to be a problem with Jakarta showing up in the Northern Hemisphere)
    for i, row in df.iterrows():
        if (row['longitude'] < xmin) or (row['longitude'] > xmax):
            df.drop(i, inplace=True)
        if (row['latitude'] < ymin) or (row['latitude'] > ymax):
            df.drop(i, inplace=True)

    return df

def getWyomingData(current_dt, stn_id, odir, incr=1, download=True):
    """
    Queries sounding data from University of Wyoming for a given station ID and datetime.
    Increment is included to allow for searching within a range around the datetime.
    If the data already exists locally it won't be downloaded again.
    :param current_dt: python datetime format
    :param stn_id: Integer. WMO station ID
    :param odir: String. Output directory
    :param incr: Integer. For the current_dt, we can query a range. This determines the size of the time window
    :param download: Boolean. If True, it will query the Wyoming site for data, if False, it will only use local data
    :return: pandas dataframe
    """

    # print(current_dt.strftime('%Y%m%d %H:%M'))
    file_name = odir + str(stn_id) + '_' + current_dt.strftime('%Y%m%dT%H%MZ') + '.txt'
    outcsv = file_name.replace('txt', 'csv')
    df = pd.DataFrame()

    url = 'http://weather.uwyo.edu/cgi-bin/sounding?region=seasia&TYPE=TEXT%3ALIST&YEAR=' + str(current_dt.year) + \
          '&MONTH=' + str(current_dt.month) + '&FROM=' + current_dt.strftime('%d%H') + \
          '&TO=' + (current_dt + dt.timedelta(hours=incr) - dt.timedelta(minutes=10)).strftime('%d%H') + \
          '&STNM=' + str(stn_id)

    if os.path.isfile(outcsv):
        print('Getting file from disk for ' + str(stn_id) + ' @ ' + current_dt.strftime('%Y%m%d %H:%M'))
        df = pd.read_csv(outcsv, parse_dates=['datetimeUTC'], dtype={'station_id':str})
    elif download:
        try:
            fn, bla = urllib.request.urlretrieve(url, file_name)
        except urllib.error.HTTPError:
            print('Unable to download due to an HTTP error')
            return df
        except:
            print('Unknown error occurred while downloading')
            return df

        data = []
        keep = False
        try:
            with open(file_name, 'r') as file:
                for line in file.readlines():
                    # print(line)
                    if '</PRE>' in line:
                        break

                    if keep:
                        data.append(line.rstrip('\n'))

                    if '<PRE>' in line:
                        keep = True
        except:
            print('unable to read the downloaded file')
            return df

        # Reads the observation time from the file
        try:
            with open(file_name, 'r') as f:
                for line in f:
                    if 'Observation time' in line:
                        obs_dt_str = line.split(': ')[1].rstrip('\n')
                        obs_dt = dt.datetime.strptime(obs_dt_str, '%y%m%d/%H%M')
        except:
            print('unable to read the obs time')
            return df

        if data:
            print('Saving sounding data from Wyoming for ' + str(stn_id) + ' @ ' + obs_dt.strftime('%Y%m%d %H:%M'))
            df = pd.DataFrame([x.split() for x in data if (not '----' in x) and (not 'hPa' in x)])
            new_header = df.iloc[0]  # grab the first row for the header
            df = df[1:]  # take the data less the header row
            df.columns = new_header  # set the header row as the df header
            df.insert(0, "datetimeUTC", obs_dt, allow_duplicates=True)
            df.datetimeUTC = pd.to_datetime(df.datetimeUTC)
            df.insert(0, "station_id", str(stn_id), allow_duplicates=True)
            df.to_csv(outcsv, index=False)

        # Delete the downloaded text file
        os.remove(file_name)
    else:
        print('No file available locally and downloads set to False')
        return df

    return df

def main(start_dt, end_dt, bbox, settings, download=True):
    """
    This function queries the WMO API "OSCAR" that stores information on locations of all types of weather station. This API allows us to query by location using the bbox that we use in other parts of this code repository. Once we have a list of stations, we then send a request to the University of Wyoming for a particular datetime and station. This is intended as a backup in case soundings are not available locally.
    :param start_dt: datetime object
    :param end_dt: datetime object
    :param bbox: list containing float values of [xmin, ymin, xmax, ymax]
    :param settings: settings read from the config file
    :return: Dictionary of 'data': pandas dataframe of the sonde data ; and 'metadata': pandas dataframe of the station metadata
    """

    # Search for sounding data at these intervals (in hours)
    # Note 1: It seems as though most sondes are reported at 00 and 12 UTC, but some local variations may occur
    # Note 2: if your start_dt is not 00UTC or 12UTC, the incr value should be 1 hour to make sure that you don't miss any sondes
    incr = 6 # Hours

    # Set up the output directory for storing the data
    odir = Path(settings['datadir']).as_posix() + '/upper-air/wyoming/'
    if not os.path.isdir(odir):
        os.makedirs(odir)

    # Get a pandas dataframe of stations (plus some metadata) for this bounding box from the WMO OSCAR API
    station_list = getUpperAirStations(bbox)

    if not isinstance(station_list, pd.DataFrame):
        return {'data': None, 'metadata': None}
    else:

        # Create an empty pandas dataframe for storing the output
        odf = pd.DataFrame()

        # Loop through all the stations found
        for i, row in station_list.iterrows():
            print(row['name'], row['territory'], row['wigosStationIdentifier'])
            stn_id = row['wigosStationIdentifier']

            # Loop through all datetimes between start and end at a frequency of the increment
            datetimes = sf.make_timeseries(start_dt, end_dt, incr)
            # current_dt = start_dt
            # while current_dt <= end_dt:
            for current_dt in datetimes:

                # This actually gets the data ...
                df = getWyomingData(current_dt, stn_id, odir, incr=incr, download=download)
                if not df.empty:
                    odf = odf.append(df, ignore_index=True)

    return {'data': odf, 'metadata': station_list}

if __name__ == '__main__':

    now = dt.datetime.utcnow()
    settings = config.load_location_settings()

    try:
        start_dt = settings['start']
    except:
        start_dt = now - dt.timedelta(days=10)

    try:
        end_dt = settings['end']
    except:
        end_dt = now

    try:
        bbox = settings['bbox']
    except:
        bbox = [100, 0, 110, 10]

    main(start_dt, end_dt, bbox, settings)
