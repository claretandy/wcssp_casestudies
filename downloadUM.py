import os, sys
import location_config as config
import datetime as dt
import std_functions as sf
from ftplib import FTP
import glob
import pdb


def get_ftp_flist(domain, event_name, settings):
    '''
    What files are currently on the ftp site for this event_name?
    :param domain: string. Either 'SEAsia' or 'Africa'
    :param event_name: string. Either 'RealTime' or '<region>/<datetime>_<event_name>'
    :param settings: dictionary of settings defined by the config file
    :return: list of filenames on the ftp site for this event_name
    '''

    if event_name == 'RealTime':
        path = '/'+domain+'/RealTime/'
    else:
        path = '/'+domain+'/CaseStudyData/' + event_name + '/'

    ftp = FTP(settings['ukmo_ftp_url'], settings['ukmo_ftp_user'], settings['ukmo_ftp_pass'])
    try:
        ftp.cwd(path)
    except:
        # If logged in as Andy, there is another directory above path ..
        path = '/WCSSP' + path
        ftp.cwd(path)

    flist = ftp.nlst()

    flist_abs = [path + fn for fn in flist]

    return flist_abs


def ftp_download_files(flist, settings):

    ftp = FTP(settings['ukmo_ftp_url'], settings['ukmo_ftp_user'], settings['ukmo_ftp_pass'])

    ofile_list = []

    for ftpfn in flist:
        path = os.path.dirname(ftpfn)
        fnbn = os.path.basename(ftpfn)
        ftp.cwd(path)
        this_dt = dt.datetime.strptime(fnbn.split('_')[0], '%Y%m%dT%H%MZ')
        if 'CaseStudyData' in ftpfn:
            event_name = path.split('CaseStudyData/')[1]
            dst_fname = settings['um_path'] + 'CaseStudyData/' + event_name + '/' + fnbn

        if 'RealTime' in ftpfn:
            model_id = fnbn.split('_')[1]
            dst_fname = settings['um_path'] + 'RealTime/' + model_id + '/' + this_dt.strftime('%Y%m/') + fnbn

        src_filesize = ftp.size(ftpfn)
        try:
            dst_filesize = os.path.getsize(dst_fname)
        except FileNotFoundError:
            dst_filesize = 0

        if not os.path.isdir(os.path.dirname(dst_fname)):
            os.makedirs(os.path.dirname(dst_fname))

        # If the filesizes don't match, then delete and re-download
        if os.path.isfile(dst_fname) and not (dst_filesize == src_filesize):
            os.remove(dst_fname)

        if not os.path.isfile(dst_fname):
            print('Downloading: ' + dst_fname)
            proses = ftp.retrbinary("RETR " + ftpfn, open(dst_fname, 'wb').write)
        else:
            print('File already exists locally and local file size agrees with ftp file: ' + dst_fname)

        if os.path.isfile(dst_fname):
            ofile_list.append(dst_fname)

    ftp.quit()

    return(ofile_list)



def get_local_flist(start, end, event_name, settings, region_type='all'):
    '''
    Gets the UM data either from a local directory or from the FTP.
    It should be possible to run this code in real time to download from the UKMO FTP site, or
    to access the model data after an event.
    This script should also cover analysis and forecast model data.
    :param start: datetime object for the start of the event
    :param end: datetime object for the end of the event
    :param event_name: string. Either 'realtime' or '<region>/<datetime>_<event_name>'
    :param settings: local settings
    :param region_type: string. Either 'all', 'event', 'region' or 'tropics'
    :return: file list
    '''

    # Approximate guess at all possible init_times
    # Removes the need for the model_id, which is not so important because
    # we are going to use glob to get all the local files
    init_times = sf.make_timeseries(start - dt.timedelta(days=5), end, 6)
    local_path = get_local_path(event_name, settings)
    first_bit = event_name.split('_')[0]
    ofilelist = []
    for it in init_times:
        if first_bit in ['realtime', 'Realtime', 'realTime', 'RealTime', 'REALTIME', 'monitoring/realtime']:
            region_type = 'all'
            search_pattern = local_path + it.strftime('%Y%m/') + it.strftime('%Y%m%dT%H%MZ') + '*'
        else:
            search_pattern = local_path + it.strftime('%Y%m%dT%H%MZ') + '*'

        it_files = sorted(glob.glob(search_pattern))
        ofilelist.extend(it_files)

    # Subset according to the region type (event, region or tropics)
    if not region_type == 'all':
        ofilelist = [fn for fn in ofilelist if region_type in fn]

    return ofilelist


def get_local_path(event_name, settings, model_id='*'):

    first_bit = event_name.split('_')[0]

    if first_bit in ['realtime', 'Realtime', 'realTime', 'RealTime', 'REALTIME', 'monitoring/realtime']:
        local_path = settings['um_path'] + 'RealTime/' + model_id + '/'
    else:
        local_path = settings['um_path'] + 'CaseStudyData/' + event_name + '/'

    return local_path


def main(start, end, bbox, event_name, organisation):
    '''
    This function is callable from the command line. It simply checks on the ftp for data relating to an event, and
    downloads it if it doesn't exist on the local datadir.
    The event_name can also be 'RealTime' in which case, the script will check the RealTime folder on the FTP for files.
    The RealTime folder should contain files from the last ~12 model runs. Older files will be deleted!
    :param start: datetime
    :param end: datetime
    :param bbox:
    :param event_name:
    :param organisation:
    :return: a list of files that are available locally following download
    '''

    settings = config.load_location_settings(organisation)

    domain = sf.getDomain_bybox(bbox)

    # What files exist locally?
    local_files = get_local_flist(start, end, event_name, settings)
    # What files exist on the FTP?
    ftp_files = get_ftp_flist(domain, event_name, settings)

    # Which files are on the ftp, but not the local directory?
    local_files_basenames = [os.path.basename(fn) for fn in local_files]
    ftp_notlocal = [ftpfn for ftpfn in ftp_files if not os.path.basename(ftpfn) in local_files_basenames]

    # Download missing files
    if ftp_notlocal:
        ftp_download_files(ftp_notlocal, settings)

    # New list of local files
    ofilelist = get_local_flist(start, end, event_name, settings)

    return ofilelist

if __name__ == '__main__':

    try:
        start_dt = dt.datetime.strptime(sys.argv[1], '%Y%m%d%H%M')
    except:
        # For realtime
        start_dt = dt.datetime.utcnow() - dt.timedelta(days=10)

    try:
        end_dt = dt.datetime.strptime(sys.argv[2], '%Y%m%d%H%M')
    except:
        # For realtime
        end_dt = dt.datetime.utcnow()

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
        # event_name = 'PeninsulaMalaysia/20200520_Johor'
        # or
        event_name = 'realtime'

    try:
        organisation = sys.argv[5]
    except:
        organisation = 'UKMO'

    main(start_dt, end_dt, event_domain, event_name, organisation)
