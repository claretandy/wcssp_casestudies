import os, sys
import datetime as dt
import location_config as config
import std_functions as sf
import shutil
import pandas as pd
import iris
import pdb

'''
This script is a collection of functions to extract data from the Met Office archives and to share that data by FTP.
It should be able to perform two main tasks:
    1. Extract case study data for a given start and end datetimes, a given spatial subset and case study name, and upload to FTP
    2. Extract model data for the whole region for the last x days, share by FTP, and delete older data
'''

def post_process(start, end, filelist, bboxes, event_name, row, settings):
    '''
    Make a spatial subset of the files in filelist, and saves into the UM/CaseStudy or UM/RealTime folder
    :param start: datetime
    :param end: datetime
    :param filelist: list of full resolution files extracted from MASS
    :param bboxes: dictionary of list of floats or integers. Tells the function which regional and/or event domains to
                use. Domains are formatted either [xmin, ymin, xmax, ymax] or None
    :param event_name: string. Format is <region_name>/<date>_<event_name>
    :param row: pandas Series. A subset taken from sf.get_default_stash_proc_codes
    :param settings: ready from the .config file
    :return: list of files in the UM/CaseStudy directory to upload to FTP
    '''

    if event_name == 'RealTime':
        # Don't do any subsetting if we're extracting RealTime data
        # Just copy the extracted data to the RealTime folder and change the filename to a nicename
        odir = settings['um_path'] + 'RealTime/'
        ofilelist = []
        for f in filelist:
            ofile = sf.make_nice_filename(os.path.basename(f))
            init_dt = dt.datetime.strptime(ofile.split('_')[0], '%Y%m%dT%H%MZ')
            model_id = ofile.split('_')[1]
            ofilepath = odir + model_id + '/' + init_dt.strftime('%Y%m/') + ofile
            if not os.path.isdir(os.path.dirname(ofilepath)):
                os.makedirs(os.path.dirname(ofilepath))
            shutil.copy(f, ofilepath)
            ofilelist.append(ofilepath)

    else:
        odir = settings['um_path'] + 'CaseStudyData/' + event_name
        if not os.path.isdir(odir):
            os.makedirs(odir)

        ofilelist = []

        for file in filelist:
            print(file)
            filebn = os.path.basename(file)
            datestr = filebn.split('_')[0]
            filestash = filebn.split('_')[-2]
            fileproc = filebn.split('_')[-1].replace('.nc', '')
            model_id = filebn.split(datestr + '_')[1].split('_' + filestash)[0]
            new_model_id = model_id.replace('_', '-')
            file_nice = file.replace(model_id, new_model_id)

            file_nice = file_nice.replace(filestash + '_' + fileproc, row.name + '_' + sf.lut_lbproc(int(fileproc)))
            ofile_base = odir + '/' + os.path.basename(file_nice)

            icube = iris.load_cube(file)

            # Loop through the dictionary of regions.
            # bboxes has 3 keys (tropics, region and event), which either contain a list of bbox coordinates or None
            # If the item contains coordinates, that means we want to subset it
            for k, val in bboxes.items():
                cube = icube.copy()
                ofile = ofile_base.replace('.nc', '_' + k + '.nc')
                if k == 'region' and val:
                    if row.levels:
                        cube = cube.extract(iris.Constraint(pressure=[925., 850., 700., 500., 200.]))
                try:
                    cube = cube.intersection(latitude=(val[1], val[3]), longitude=(val[0], val[2]))
                    cube = sf.periodConstraint(cube, start, end)
                    iris.save(cube, ofile, zlib=True)
                    ofilelist.append(ofile)
                except TypeError:
                    continue
                except:
                    print('File either outside domain or time constraints')

    return ofilelist

def domain_size_decider(row, model_id, regbbox, eventbbox, event_name):
    '''
    Decides, using the stashdf row (from std_stashcodes.csv), whether we are subsetting using the event_domain, the
    regional bounding box or the global tropics
    :param row: pandas Series. A subset taken from sf.get_default_stash_proc_codes
    :param model_id: string. The model identifier. Could be anyone of 'ga6', 'ga7', 'km4p4', 'indkm1p5', 'malkm1p5',
            'phikm1p5', 'global_prods' (global operational), 'africa_prods' (africa operational)
    :param regbbox: list of floats. Bounding box of the (larger) regional domain. Contains [xmin, ymin, xmax, ymax]
    :param eventbbox: list of floats. Bounding box of the (smaller) event domain. Contains [xmin, ymin, xmax, ymax]
    :return: dictionary of domains to use for subsetting. If a domain is set to None, then it is not used for this
            stash code / region combination
    '''

    # Models that have a full global coverage
    global_models = ['analysis', 'global_prods', 'opfc']
    # Models that are likely to have a full regional coverage
    regional_models = ['analysis', 'ga6', 'ga7', 'km4p4', 'global_prods', 'africa_prods', 'opfc']
    # No point sharing global data not in the tropics, and this reduces the data size by 2/3
    tropicsbbox = [-180, -30, 180, 30]

    if row.share_list:

        if event_name == 'RealTime':
            '''
            If the event_name is 'RealTime', then we export the regional domain, but only leave the data on the
            FTP for a limited time
            '''
            tropics = tropicsbbox if row.share_tropics and model_id in global_models else None
            region = regbbox if model_id in regional_models else None
            event = None

        else:
            '''
            If the event_name is not 'RealTime', then we assume it is an event.
            The logic below will lead to some duplication, but I think this is acceptable because it reduces 
            complexity in the plotting scripts
            '''
            tropics = tropicsbbox if row.share_tropics and model_id in global_models else None
            region = regbbox if row.share_region and model_id in regional_models else None
            event = eventbbox

    return {'tropics': tropics, 'region': region, 'event': event}


def main(start, end, event_domain, event_name):

    '''

    :param start: datetime
    :param end: datetime
    :param event_domain: list of floats. Bounding box of the (smaller) event domain. Contains [xmin, ymin, xmax, ymax]
    :param event_name: string. Format is <region_name>/<date>_<event_name>
    :return: Nothing. Data extracted to:
                - /scratch for realtime or full model fields
                - /data/users for casestudy data that might be worth keeping for longer
                - All data sent to FTP
    '''

    # Set some location-specific defaults
    settings = config.load_location_settings('UKMO')

    # Checks for common spelling mistakes
    if event_name in ['realtime', 'Realtime', 'realTime', 'RealTime', 'REALTIME']:
        event_name = 'RealTime'

    # Gets all the stash codes tagged as share
    stashdf = sf.get_default_stash_proc_codes(list_type='share')

    # Gets the large scale domain name (either 'SEAsia', 'Africa', or 'global')
    domain = sf.getDomain_bybox(event_domain)
    regbbox = sf.getBBox_byRegionName(domain)
    model_ids = sf.getModels_bybox(event_domain)['model_list']

    # Note that the event_name follows the format region/casestudy_name
    if event_name == 'RealTime':
        ftp_path = '/WCSSP/SEAsia/RealTime/'
        remove_old = True
        # km1p5 data is too big to send in realtime
        model_ids = [mi for mi in model_ids if not 'km1p5' in mi]
    else:
        ftp_path = '/WCSSP/SEAsia/CaseStudyData/' + event_name
        remove_old = False

    for row in stashdf.itertuples(index=False):

        # Get the UM analysis data
        bboxes = domain_size_decider(row, 'analysis', regbbox, event_domain, event_name)
        ana_start = start - dt.timedelta(days=5)
        filelist_analysis = sf.selectAnalysisDataFromMass(ana_start, end, row.stash, lbproc=row.lbproc, lblev=row.levels)
        filelist_analysis = post_process(ana_start, end, filelist_analysis, bboxes, event_name, row, settings)
        sf.send_to_ftp(filelist_analysis, ftp_path, settings, removeold=remove_old)

        # Get the UM model data
        for model_id in model_ids:
            init_times = sf.getInitTimes(start, end, domain, model_id=model_id)
            bboxes = domain_size_decider(row, model_id, regbbox, event_domain, event_name)
            filelist_models = sf.selectModelDataFromMASS(init_times, row.stash, lbproc=row.lbproc, lblev=row.levels, plotdomain=event_domain, searchtxt=model_id)
            filelist_models = post_process(start, end, filelist_models, bboxes, event_name, row, settings)
            sf.send_to_ftp(filelist_models, ftp_path, settings, removeold=remove_old)


if __name__ == '__main__':

    # extractUM.py ${start} ${end} ${event_domain} ${eventname}

    try:
        start_dt = dt.datetime.strptime(sys.argv[1], '%Y%m%d%H%M')
    except:
        # For testing
        start_dt = dt.datetime.now() - dt.timedelta(days=10)

    try:
        end_dt = dt.datetime.strptime(sys.argv[2], '%Y%m%d%H%M')
    except:
        # For testing
        end_dt = dt.datetime.now()

    try:
        domain_str = sys.argv[3]
        event_domain = [float(x) for x in domain_str.split(',')]
    except:
        # For testing
        event_domain = [90, -10, 120, 20]

    try:
        event_name = sys.argv[4]
    except:
        # For testing
        event_name = 'RealTime'

    main(start_dt, end_dt, event_domain, event_name)