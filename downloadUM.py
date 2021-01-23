import os, sys
import location_config as config
import datetime as dt
import std_functions as sf
from ftplib import FTP
import glob
import pdb


def get_ftp_flist(domain, event_name, settings):
    '''
    What files are currently on the ftp site for this region_name?
    :param domain: string. Either 'SEAsia' or 'Africa'
    :param event_name: string. Either 'RealTime' or '<region>/<datetime>_<region_name>'
    :param settings: dictionary of settings defined by the config file
    :return: list of filenames on the ftp site for this region_name
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

        dst_fname = settings['um_path'] + settings['region_name'] + '/' + settings['location_name'] + '/' + this_dt.strftime('%Y%m/') + fnbn
        # if 'CaseStudyData' in ftpfn:
        #     event_name = path.split('CaseStudyData/')[1]
        #     dst_fname = settings['um_path'] + 'CaseStudyData/' + event_name + '/' + fnbn
        #
        # if 'RealTime' in ftpfn:
        #     model_id = fnbn.split('_')[1]
        #     dst_fname = settings['um_path'] + 'RealTime/' + model_id + '/' + this_dt.strftime('%Y%m/') + fnbn

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



def get_local_flist(start, end, settings, region_type='all'):
    '''
    Gets the UM data either from a local directory or from the FTP.
    It should be possible to run this code in real time to download from the UKMO FTP site, or
    to access the model data after an event.
    This script should also cover analysis and forecast model data.
    :param start: datetime object for the start of the event
    :param end: datetime object for the end of the event
    :param region_name: string. Name for larger region
    :param location_name. string. Name for a zoom within the region
    :param settings: local settings
    :param region_type: string. Either 'all', 'event', 'region' or 'tropics'
    :return: file list
    '''

    # Approximate guess at all possible init_times
    # Removes the need for the model_id, which is not so important because
    # we are going to use glob to get all the local files
    init_times = sf.make_timeseries(start - dt.timedelta(days=5), end, 6)
    local_path = settings['um_path'] + '/' + settings['region_name'] + '/' + settings['location_name'] + '/'

    ofilelist = []
    for it in init_times:
        search_pattern = local_path + it.strftime('%Y%m') + '/' + it.strftime('%Y%m%dT%H%MZ') + '*'
        try:
            it_files = sorted(glob.glob(search_pattern))
            ofilelist.extend(it_files)
        except:
            pass

    # Subset according to the region type (event, region or tropics)
    if not region_type == 'all':
        ofilelist = [fn for fn in ofilelist if region_type in fn]

    return ofilelist


def main(start, end, bbox, settings):
    '''
    This function is callable from the command line. It simply checks on the ftp for data relating to an event, and
    downloads it if it doesn't exist on the local datadir.
    The region_name can also be 'RealTime' in which case, the script will check the RealTime folder on the FTP for files.
    The RealTime folder should contain files from the last ~12 model runs. Older files will be deleted!
    :param start: datetime
    :param end: datetime
    :param bbox:
    :param settings:
    :return: a list of files that are available locally following download
    '''

    domain = sf.getModelDomain_bybox(bbox)

    # What files exist locally?
    local_files = get_local_flist(start, end, settings)
    # What files exist on the FTP?
    ftp_files = get_ftp_flist(domain, settings)

    # Which files are on the ftp, but not the local directory?
    local_files_basenames = [os.path.basename(fn) for fn in local_files]
    ftp_notlocal = [ftpfn for ftpfn in ftp_files if not os.path.basename(ftpfn) in local_files_basenames]

    # Download missing files
    if ftp_notlocal:
        ftp_download_files(ftp_notlocal, settings)

    # New list of local files
    ofilelist = get_local_flist(start, end, settings)

    return ofilelist

if __name__ == '__main__':

    try:
        organisation = os.environ['organisation']
    except:
        organisation = 'UKMO'

    settings = config.load_location_settings(organisation)
    main(settings['start'], settings['end'], settings['bbox'], organisation)
