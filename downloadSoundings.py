import os, sys
import location_config as config
import datetime as dt
import urllib.request
import numpy as np
import std_functions as sf
from ftplib import FTP
import glob
import iris
import location_config as config
from pathlib import Path
import pandas as pd


def main(start_dt, end_dt, stn_id, settings):
    # do stuff

    odir = Path(settings['datadir']).as_posix() + '/upper-air/wyoming/'
    odf = []
    if not os.path.isdir(odir):
        os.makedirs(odir)
    incr = 1 # hour
    current_dt = start_dt
    while current_dt <= end_dt:
        print(current_dt)
        url = 'http://weather.uwyo.edu/cgi-bin/sounding?region=seasia&TYPE=TEXT%3ALIST&YEAR=' + str(current_dt.year) +'&MONTH='+str(current_dt.month)+'&FROM='+(current_dt).strftime('%d%H')+'&TO='+(current_dt + dt.timedelta(hours=incr) - dt.timedelta(minutes=10)).strftime('%d%H')+'&STNM='+str(stn_id)
        file_name = odir + str(stn_id) + '_' + current_dt.strftime('%Y%m%dT%H%MZ') + '.txt'
        fn, bla = urllib.request.urlretrieve(url, file_name)
        # Check if there is any data in the file - if not, delete
        data = []

        keep = False
        with open(file_name, 'r') as file:
            for line in file.readlines():
                if '</PRE>' in line:
                    break

                if keep:
                    data.append(line.rstrip('\n'))

                if '<PRE>' in line:
                    keep = True

        if data:
            print(data)
            df = pd.DataFrame([x.split() for x in data if (not '----' in x) and (not 'hPa' in x)])
            new_header = df.iloc[0]  # grab the first row for the header
            df = df[1:]  # take the data less the header row
            df.columns = new_header  # set the header row as the df header
            df.insert(0, "datetimeUTC", current_dt, allow_duplicates=True)
            df.to_csv(file_name.replace('txt','csv'))

        if isinstance(odf, list):
            odf = df.copy()
        else:
            odf.append(df)

        # Delete the downloaded file
        os.remove(file_name)

        current_dt += dt.timedelta(hours=incr)

    return odf

if __name__ == '__main__':

    start_dt = dt.datetime(sys.argv[1], '%Y%m%dT%H%MZ')
    end_dt = dt.datetime(sys.argv[2], '%Y%m%dT%H%MZ')
    stn_id = sys.argv[3]
    organisation = sys.argv[4]

    settings = config.load_location_settings(organisation)

    main(start_dt, end_dt, stn_id, settings)