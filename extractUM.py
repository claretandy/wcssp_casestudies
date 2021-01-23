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

def post_process(start, end, filelist, bboxes, row, settings):
    '''
    Make a spatial subset of the files in filelist, and saves into the UM/CaseStudy or UM/RealTime folder
    :param start: datetime
    :param end: datetime
    :param filelist: list of full resolution files extracted from MASS
    :param bboxes: dictionary of list of floats or integers. Tells the function which regional and/or event domains to
                use. Domains are formatted either [xmin, ymin, xmax, ymax] or None
    :param event_name: string. Format is <region_name>/<date>_<region_name>
    :param row: pandas Series. A subset taken from sf.get_default_stash_proc_codes
    :param settings: ready from the .config file
    :return: list of files in the UM/CaseStudy directory to upload to FTP
    '''

    odir = settings['um_path'].rstrip('/') + '/' + settings['region_name'] + '/' + settings['location_name']
    ofilelist = []
    for f in filelist:
        ofile = sf.make_nice_filename(os.path.basename(f))
        init_dt = dt.datetime.strptime(ofile.split('_')[0], '%Y%m%dT%H%MZ')
        model_id = os.path.basename(f).split('_')[1]
        ofilepath = odir + '/' + init_dt.strftime('%Y%m/') + ofile

        if not os.path.isdir(os.path.dirname(ofilepath)):
            os.makedirs(os.path.dirname(ofilepath))

        # Loop through the dictionary of regions.
        # bboxes has 3 keys (tropics, region and event), which either contain a list of bbox coordinates or None
        # If the item contains coordinates, that means we want to subset it
        try:
            icube = iris.load_cube(f)
        except:
            os.remove(f)
            continue

        for k, val in bboxes.items():

            if val == 'tropics' and ('ga' in model_id or 'global' in model_id):
                # No need to process the global model at tropics scale
                continue

            if val:
                cube = icube.copy()
                ofile = ofilepath.replace('.nc', '_' + k + '.nc')
                print('Saving:',ofile)
                if k == 'region':
                    if row.levels:
                        cube = cube.extract(iris.Constraint(pressure=[925., 850., 700., 500., 200.]))
                try:
                    cube = cube.intersection(latitude=(val[1], val[3]), longitude=(val[0], val[2]))
                    cube = sf.periodConstraint(cube, start, end, greedy=True)
                    iris.save(cube, ofile, zlib=True)
                    ofilelist.append(ofile)
                except TypeError:
                    continue
                except:
                    print('   File either outside bbox or time constraints')

    return ofilelist

def domain_size_decider(row, model_id, regbbox, eventbbox):
    '''
    Decides, using the stashdf row (from std_stashcodes.csv), whether we are subsetting using the bbox, the
    regional bounding box or the global tropics
    :param row: pandas Series. A subset taken from sf.get_default_stash_proc_codes
    :param model_id: string. The model identifier. Could be anyone of 'ga6', 'ga7', 'km4p4', 'indkm1p5', 'malkm1p5',
            'phikm1p5', 'global_prods' (global operational), 'africa_prods' (africa operational)
    :param regbbox: list of floats. Bounding box of the (larger) regional bbox. Contains [xmin, ymin, xmax, ymax]
    :param eventbbox: list of floats. Bounding box of the (smaller) event bbox. Contains [xmin, ymin, xmax, ymax]
    :return: dictionary of domains to use for subsetting. If a bbox is set to None, then it is not used for this
            stash code / region combination
    '''

    # Models that have a full global coverage
    global_models = ['analysis', 'global_prods', 'opfc']
    # Models that are likely to have a full regional coverage
    regional_models = ['analysis', 'ga6', 'ga7', 'km4p4', 'global_prods', 'africa_prods', 'opfc', 'psuite42']
    # No point sharing global data not in the tropics, and this reduces the data size by 2/3
    tropicsbbox = [-180, -30, 180, 30]

    try:
        tropics = tropicsbbox if row.share_tropics and model_id in global_models else None
        region = regbbox if row.share_region and model_id in regional_models else None
        event = eventbbox if row.share_event else None
    except:
        tropics = None
        region = None
        event = None

    return {'tropics': tropics, 'region': region, 'event': event}


def main(start, end, bbox, event_name, settings, model_ids=None, stash_colname='share_event', ftp_upload=False):

    '''

    :param start: datetime
    :param end: datetime
    :param bbox: list of floats. Bounding box of the (smaller) event bbox. Contains [xmin, ymin, xmax, ymax]
    :param event_name: string. Format is <region_name>/<date>_<region_name>
    :return: Nothing. Data extracted to:
                - /scratch for realtime or full model fields
                - /data/users for casestudy data that might be worth keeping for longer
                - All data sent to FTP
    '''

    # Checks for common spelling mistakes
    if any(['RealTime' if x in event_name else None for x in ['realtime', 'Realtime', 'realTime', 'RealTime', 'REALTIME']]):
        event_name = 'RealTime'

    # Gets all the stash codes tagged as share
    stashdf = sf.get_default_stash_proc_codes(list_type=stash_colname)

    # Gets the large scale bbox name (either 'SEAsia', 'Africa', or 'global')
    domain = sf.getModelDomain_bybox(bbox)
    regbbox = sf.getBBox_byRegionName(domain)
    if not model_ids:
        model_ids = sf.getModels_bybox(bbox)['model_list']

    # Set ftp path etc
    ftp_path = '/WCSSP/'+event_name+'/'
    remove_old = False
    # km1p5 data is too big to send in realtime
    model_ids = [mi for mi in model_ids if not 'km1p5' in mi]

    # if event_name == 'RealTime':
    #     ftp_path = '/WCSSP/SEAsia/RealTime/'
    #     remove_old = True
    #     # km1p5 data is too big to send in realtime
    #     model_ids = [mi for mi in model_ids if not 'km1p5' in mi]
    # else:
    #     ftp_path = '/WCSSP/SEAsia/CaseStudyData/' + event_name
    #     remove_old = False

    for row in stashdf.itertuples(index=False):

        # Get the UM analysis data
        bboxes = domain_size_decider(row, 'analysis', regbbox, bbox)
        ana_start = start - dt.timedelta(days=1)
        filelist_analysis = sf.selectAnalysisDataFromMass(ana_start, end, row.stash, lbproc=row.lbproc, lblev=row.levels)
        filelist_analysis = post_process(ana_start, end, filelist_analysis, bboxes, row, settings)
        if ftp_upload:
            sf.send_to_ftp(filelist_analysis, ftp_path, settings, removeold=remove_old)

        # Get the UM model data
        for model_id in model_ids:
            print(model_id, row.stash, row.lbproc)
            init_times = sf.getInitTimes(start, end, domain, model_id=model_id)
            bboxes = domain_size_decider(row, model_id, regbbox, bbox)
            filelist_models = sf.selectModelDataFromMASS(init_times, row.stash, lbproc=row.lbproc, lblev=row.levels, domain=sf.getModelDomain_bybox(bbox), plotdomain=bbox, modelid_searchtxt=model_id)
            filelist_models = post_process(start, end, filelist_models, bboxes, row, settings)
            if ftp_upload:
                sf.send_to_ftp(filelist_models, ftp_path, settings, removeold=remove_old)


if __name__ == '__main__':

    try:
        organisation = os.environ['organisation']
    except:
        organisation = 'UKMO'

    settings = config.load_location_settings(organisation)

    start_dt = settings['start']
    end_dt = settings['end']
    bbox = settings['bbox']
    event_name = settings['region_name']+'/'+settings['location_name']
    model_ids = settings['model_ids']
    stash_colname = settings['stash_colname']
    ftp_upload = settings['ftp_upload']

    main(start_dt, end_dt, bbox, event_name, settings, model_ids=model_ids, stash_colname=stash_colname, ftp_upload=ftp_upload)
