import os, sys
import location_config as config
from ftplib import FTP
import datetime as dt

def getUM(start, end, dataname, settings):

    # Check local

def checkLocal():

def main(start, end, organisation):
    # Do some downloading
    settings = config.load_location_settings(organisation)
    umdir = settings['datadir'].rstrip('/') + '/UM/'
    if not os.path.isdir(umdir):
        os.makedirs(umdir)

    datalist = ['analysis', 'ga7', 'km4p4', 'km1p5']
    for dataname in datalist:

        local_datadir = umdir.rstrip('/') + dataname
        if not os.path.isdir(local_datadir):
            os.makedirs(local_datadir)

        cubelist = getUM(start, end, dataname, settings)



if __name__ == '__main__':

    now = dt.datetime.utcnow()

    try:
        start = dt.datetime.strptime(sys.argv[2][:8], "%Y%m%d")  # Needs to be formatted YYYYMMDD
    except IndexError:
        start = now.date() - dt.timedelta(days=7)

    try:
        end = dt.datetime.strptime(sys.argv[3][:8], "%Y%m%d")  # Needs to be formatted YYYYMMDD
    except IndexError:
        end = now.date()

    organisation = sys.argv[3]

    main(start, end, organisation)
