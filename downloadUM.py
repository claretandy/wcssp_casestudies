import os, sys
import location_config as config
from ftplib import FTP
import datetime as dt
import std_functions as sf

def getUM(start, end, model_id, var, settings, returnCube=False):
    '''
    Gets the UM data either from a local directory or from the FTP.
    It should be possible to run this code in real time to download from the UKMO FTP site, or
    to access the model data after an event.
    This script should also cover analysis and forecast model data.
    :param start: datetime object for the start of the event
    :param end: datetime object for the end of the event
    :param model_id: choose from [analysis|ga7|km4p4|km1p5]
    :param var: which model variable do we want?
    :param settings: local settings
    :param returnCube: [True|False] - do we want to return a cube, or just download the data?
    :return:
    '''

    if model_id == 'analysis':
        cubelist = getUM_analysis(start, end, var, settings, returnCube=False)
    else:
        cubelist = getUM_models(start, end, model_id, var, settings, returnCube=False)

    return cubelist

def getUM_analysis(start, end, var, settings, returnCube=False):

    analysis_incr = 6 # hours

    #########
    # Change nothing from here onwards

    # Directory to save global analysis data to
    if not odir:
        odir = '/scratch/hadhy/ModelData/UM_Analysis/'

    # Times of day for which the analysis is run
    analysis_times = np.arange(start=0, stop=24, step=analysis_incr)

    # Set lblevs if not already done so
    if lblev:
        lblev = [10, 15, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 750, 850, 900, 925, 950, 1000]

    # Set the output path
    if not pathlib.Path(odir).is_dir():
        pathlib.Path(odir).mkdir(parents=True)

    # Set the output cubelist if required
    if returncube:
        ocubes = iris.cube.CubeList([])

    ofilelist = []

    # Make sure the start and end times occur when we should have an analysis
    if not start_dt.hour in analysis_times:
        while not start_dt.hour in analysis_times:
            start_dt -= dt.timedelta(hours=1)

    if not end_dt.hour in analysis_times:
        while not end_dt.hour in analysis_times:
            end_dt += dt.timedelta(hours=1)

    # Loop through all analysis times
    this_dt = start_dt
    while this_dt <= end_dt:

        it = this_dt.strftime('%Y%m%dT%H%MZ')
        ofile = pathlib.PurePath(odir, it + '_analysis_' + str(stash) + '_' + str(lbproc) + '.nc').as_posix()
        if (not os.path.isfile(ofile)) or overwrite:
            # Try to download from FTP
        else:

    return cubelist

def getUM_model(start, end, model_id, var, settings, returnCube=False):
    # For this date range, what model initialisation times do we want?
    init_times = sf.getInitTimes(start, end, 'SEAsia', model_id=model_id)
    # Build a filename

    # Check local

    return cubelist


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
