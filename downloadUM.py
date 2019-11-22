import os, sys
import location_config as config
from ftplib import FTP
import datetime as dt
import std_functions as sf

def getUM(start, end, model_id, var, settings, returnCube=False):

    sf.getInitTimes(start, end, domain, model_id=model_id)
    # Build a filename

    # Check local

def checkLocal():


def main(start, end, organisation):
    # Do some downloading
    settings = config.load_location_settings(organisation)
    umdir = settings['um_path']

    model_list = ['analysis', 'ga7', 'km4p4', 'km1p5']
    for model_id in model_list:

        local_datadir = umdir.rstrip('/') + model_id
        if not os.path.isdir(local_datadir):
            os.makedirs(local_datadir)

        cubelist = getUM(start, end, model_id, var, settings, returnCube=False)



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
