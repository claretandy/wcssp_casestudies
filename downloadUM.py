import os, sys
import location_config as config
import datetime as dt
import numpy as np
import std_functions as sf
from ftplib import FTP
import glob
import iris
import pdb

def get_ftp_flist(model_id, settings):

    if 'km1p5' in model_id:
        path = '/' + settings['country'] + '/'
    else:
        path = '/SEAsia/'
    ftp = FTP(settings['ukmo_ftp_url'], settings['ukmo_ftp_user'], settings['ukmo_ftp_pass'])
    ftp.cwd(path)
    flist = ftp.nlst()

    return flist


def ftp_download_files(flist, model_id, settings):

    if model_id == 'km1p5':
        path = '/' + settings['country'] + '/'
    else:
        path = '/SEAsia/'

    ftp = FTP(settings['ukmo_ftp_url'], settings['ukmo_ftp_user'], settings['ukmo_ftp_pass'])
    ftp.cwd(path)
    dataloc = settings['um_path'] + model_id

    for filename in flist:

        src_fname = path + filename
        this_dt = dt.datetime.strptime(filename.split('_')[0], '%Y%m%dT%H%MZ')
        dst_fname = dataloc + '/' + this_dt.strftime('%Y%m') + '/' + filename

        if not os.path.isdir(os.path.dirname(dst_fname)):
            os.makedirs(os.path.dirname(dst_fname))

        if not os.path.isfile(dst_fname):
            print('Downloading: ' + dst_fname)
            proses = ftp.retrbinary("RETR " + src_fname, open(dst_fname, 'wb').write)
        else:
            print('Nothing to do: ' + src_fname)

    ftp.quit()


def get_analysis_times(start, end, incr):
    '''
    Return a datetime timeseries between the start and end dates, increasing by the specified increment
    :param start: datetime
    :param end: datetime
    :param incr: int (hours)
    :return: list of datetimes
    '''

    ts = []
    current_dt = start
    while current_dt <= end:
        ts.append(current_dt)
        current_dt += dt.timedelta(hours=incr)

    return ts


def expand_start_end(start_dt, end_dt, model_id):
    '''
    Since the analysis runs every 6 hours, we may want to expand the start and end times to get one
    earlier and one later analysis runs
    :param start_dt: datetime object set by runall.sh
    :param end_dt: datetime object set by runall.sh
    :param model_id: String, can be one of ['analysis', 'ga7', 'km4p4', 'km1p5']
    :return: list of start and end datetimes
    '''

    if model_id == 'analysis':
        incr = 6  # hours
    else:
        incr = 12

    times = np.arange(start=0, stop=24, step=incr)

    # Make sure the start and end times occur when we should have an analysis
    if not start_dt.hour in times:
        while not start_dt.hour in times:
            start_dt -= dt.timedelta(hours=1)

    if not end_dt.hour in times:
        while not end_dt.hour in times:
            end_dt += dt.timedelta(hours=1)

    return [start_dt, end_dt]


def getVarList(model_id):
    '''
    Gets the variables that are known to exist for a given model_id
    :param model_id: string; can be either ['analysis', 'ga7', 'km4p4', 'km1p5']
    :return: list of variable names that can be used to create filename strings
    '''

    var_dict = {
        'analysis' : [('rh_wrt_water_levels', 'inst'), ('templevels', 'inst'), ('Vwind10m', 'inst'), ('Vwind', 'inst'), ('Uwind10m', 'inst'), ('Uwind', 'inst'), ('temp1.5m', 'inst'), ('dewpttemp1.5m', 'inst'), ('precip', '3hr'), ('precip', 'inst')],
        'ga7' : [(5216, 0), (5216, 128), (15201, 0), (15202, 0), (16202, 0), (16203, 0), (16205, 0), (30205, 0), (30461, 0)],
        'km4p4' : [(4203, 0), (4203, 128), (15201, 0), (15202, 0), (16202, 0), (16203, 0)],
        'km1p5' : []
    }
    return var_dict[model_id]


def stashLUT(sc, style='short'):
    '''
    Gets the description of the stash code either in long or short format
    :param sc: if is an integer, returns the description, BUT, if sc is a string, returns a list of stash code numbers
    :param style: string. Can be: either 'short' or 'long'
    :return: either a string of the stash code name, or a list of matching stash codes
    '''


    # TODO Add to this list when new model data is added
    stashcodes = {
        3225: {'short': 'Xwind-10m', 'long': '10m X Wind'},
        3226: {'short': 'Ywind-10m', 'long': '10m Y Wind'},
        3236: {'short': 'temp-1.5m', 'long': '1.5m Air Temperature'},
        4203: {'short': 'precip', 'long': 'Precipitation'},
        5216: {'short': 'precip', 'long': 'Precipitation'},
        15201: {'short': 'Xwind', 'long': 'X Wind on levels'},
        15202: {'short': 'Ywind', 'long': 'Y Wind on levels'},
        16202: {'short': 'geopot-ht', 'long': 'Geopotential Height on levels'},
        16203: {'short': 'temp-levels', 'long': 'Temperature on levels'},
        16205: {'short': 'wetbulbpot-levels', 'long': 'Wet Bulb Potential Temperature on levels'},
        30205: {'short': 'spechumidity-levels', 'long': 'Specific Humidity on levels'},
        30461: {'short': 'totcolumnwatervap', 'long': 'Total Column Water Vapour'}
    }

    if isinstance(sc, int):
        return stashcodes[sc][style]

    if isinstance(sc, str):
        return [stitems[0] for val, stitems in zip(stashcodes.values(), stashcodes.items()) if sc in val['long'].lower()]


def getUM(start, end, model_id, settings):
    '''
    Gets the UM data either from a local directory or from the FTP.
    It should be possible to run this code in real time to download from the UKMO FTP site, or
    to access the model data after an event.
    This script should also cover analysis and forecast model data.
    :param start: datetime object for the start of the event
    :param end: datetime object for the end of the event
    :param model_id: choose from [analysis|ga7|km4p4|km1p5]
    :param settings: local settings
    :return:
    '''

    if model_id == 'analysis':
        incr = 6 # Hours
        start1, end1 = expand_start_end(start, end, model_id)
        init_times = get_analysis_times(start1, end1, incr)
    else:
        init_times = sf.getInitTimes(start, end, 'SEAsia', model_id, sf.get_fc_length(model_id), [0,12])

    # Get FTP file list
    flist = get_ftp_flist(model_id, settings)

    # Subset to match this model_id
    flist = [fl for fl in flist if model_id in fl]

    # Subset to match init_times
    tf = '%Y%m%dT%H%MZ'
    init_times_txt = [it.strftime(tf) for it in init_times]
    init_times_avail = sorted(list(set([dt.datetime.strptime(fl.split('_')[0], tf) for fl in flist if fl.split('_')[0] in init_times_txt])))
    flist_init_times = [fl for fl in flist if dt.datetime.strptime(fl.split('_')[0], tf) in init_times_avail]

    # Download all available data for this flist and init times that we want
    if flist_init_times:
        ftp_download_files(flist_init_times, model_id, settings)
        return flist_init_times
    else:
        return 'No files on the FTP available for ' + model_id + ' for period ' + start.strftime('%Y-%m-%d %H:%M') + ' to ' + end.strftime('%Y-%m-%d %H:%M')


def loadUM(start, end, model_id, bbox, settings, var='all'):
    '''
    Loads the UM data for the specified period, model_id, variables and subsets by bbox
    :param start: datetime object
    :param end: datetime object
    :param model_id: string. Select from ['analysis', 'ga7', 'km4p4', 'km1p5']
    :param bbox: dictionary specifying bounding box that we want to plot
            e.g. {'xmin': 99, 'ymin': 0.5, 'xmax': 106, 'ymax': 7.5}
    :param settings: settings from the config file
    :param var: Either not specified (i.e. 'all') or a string or a list of strings
            e.g. ['wind', 'temperature']
    :return: Cubelist of all variables and init_times
    '''

    full_file_list = getUM(start, end, model_id, settings)

    file_list = []
    if not var == 'all':
        if isinstance(var, str):
            var = [var]
        for file in full_file_list:
            for v in var:
                if isinstance(v, int):
                    v = stashLUT(v)
                #
                # [val['short'] for val, sc in zip(stashcodes.values(), stashcodes.items()) if v in val['short']]
                if v.lower() in file.lower():
                    file_list.append(file)


    for file in file_list:
        cube = iris.load_cube(file)


def checkUM_availability(start, end, model_id, settings, var='all'):
    '''
    Checks what variables are available locally and on the ftp for the model_id and period.
    Can accept either a list of variables or if not set, will list all vars available locally and online
    :param start: datetime object
    :param end: datetime object
    :param model_id: string. Select from ['analysis', 'ga7', 'km4p4', 'km1p5']
    :param settings: settings from the config file
    :param var: Either 'all' or a string or a list of strings
            e.g. ['wind', 'temperature']
    :return: Nothing, but prints out a nice list of the files held locally and additional files available on the FTP
    '''

    if model_id == 'analysis':
        incr = 6 # Hours
        start1, end1 = expand_start_end(start, end, model_id)
        init_times = get_analysis_times(start1, end1, incr)
    else:
        init_times = sf.getInitTimes(start, end, 'SEAsia', model_id, sf.get_fc_length(model_id), [0,12])

    # FTP files ....
    # Get FTP file list
    flist = get_ftp_flist(model_id, settings)
    # Subset to match this model_id
    flist = [fl for fl in flist if model_id in fl]
    # Subset to match init_times
    tf = '%Y%m%dT%H%MZ'
    init_times_txt = [it.strftime(tf) for it in init_times]
    init_times_avail = sorted(list(set([dt.datetime.strptime(fl.split('_')[0], tf) for fl in flist if fl.split('_')[0] in init_times_txt])))
    flist_init_times = [fl for fl in flist if dt.datetime.strptime(fl.split('_')[0], tf) in init_times_avail]

    # Local files ...
    #  Get local file list
    local_path = settings['um_path'] + model_id + '/'
    flist_local = [x for it in init_times for x in glob.glob(local_path + it.strftime('%Y%m') + '/' + it.strftime('%Y%m%dT%H%MZ') + '*')]

    ftp_files = []
    local_files = []
    if not var == 'all':
        if isinstance(var, str):
            var = [var]
        for v in var:
            # Do something with local files
            for lfile in flist_local:
                lstash = lfile.split(get_full_model_id(model_id))[1].split('_')[1]
                try:
                    # If it's a number, this will work
                    lstash = stashLUT(int(lstash), style='long')
                except ValueError:
                    #  If it's already a descriptive stash name, pass
                    pass
                if v.lower() in lstash.lower():
                    #  If our search phrase is in the descriptive stash name, add the file to the list
                    local_files.append(lfile)

            # Do something with ftp files
            for file in flist_init_times:
                stash = file.split(get_full_model_id(model_id))[1].split('_')[1]
                try:
                    # If it's a number, this will work
                    stash = stashLUT(int(stash), style='long')
                except ValueError:
                    # If it's already a descriptive stash name, pass
                    pass
                if v.lower() in stash.lower():
                    # If our search phrase is in the descriptive stash name, add the file to the list
                    ftp_files.append(file)
    else:
        ftp_files.extend(flist_init_times)
        local_files.extend(flist_local)

    # How many files on the ftp are also saved locally, and how many aren't?
    local_files = [os.path.basename(lf) for lf in local_files]
    ftp_and_local = [ff for ff in ftp_files if ff in local_files]
    print('Local file count: ' + str(len(local_files)) + ' | FTP file count: ' + str(len(ftp_files)))
    if len(local_files) < len(ftp_files):
        extra = [ff for ff in ftp_files if ff not in local_files]
        print('   Extra files to download from FTP: ', extra)

    # Of all the possible files, how many are not saved locally AND not on the ftp?
    pot_file_list = []
    if not var == 'all':
        for it in init_times:
            for v in var:
                for st, lb in getVarList(model_id):
                    try:
                        # If it's a number, this will work
                        st = stashLUT(int(st), style='short')
                    except ValueError:
                        #  If it's already a descriptive stash name, pass
                        pass
                    if v.lower() in st.lower():
                        #  If our search phrase is in the descriptive stash name, add the file to the list
                        pot_file = it.strftime(tf) + '_' + get_full_model_id(model_id) + '_' + str(st) + '_' + str(
                            lb) + '.nc'
                        pot_file_list.append(pot_file)

    print('Potentially there are ' + str(len(pot_file_list) - len(local_files)) + ' more files available')


def get_full_model_id(model_id):

    # TODO : Check km1p5 name
    mod_dict = {
        'analysis' : 'analysis',
        'ga7' : 'SEA5_n1280_ga7',
        'km4p4' : 'SEA5_km4p4_ra2t',
        'km1p5' : 'SEA5_km1p5_ra2t'
    }

    return mod_dict[model_id]

def main(start, end, organisation):
    # Do some downloading
    settings = config.load_location_settings(organisation)

    modelcheck = ['analysis', 'ga7', 'km4p4', 'km1p5']
    for model_id in modelcheck:

        filelist = getUM(start, end, model_id, settings)


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
