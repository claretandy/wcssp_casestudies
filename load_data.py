import glob
import os, sys
import datetime as dt
import iris
import numpy as np
import numpy.ma as ma
import cf_units
import location_config as config
from iris.experimental.equalise_cubes import equalise_attributes
import std_functions as sf
from downloadUM import get_local_flist
from extractSatellite import getAutoSatDetails
import downloadSoundings
import downloadGPM
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import pdb


def wyoming_soundings(start_dt, end_dt, bbox, settings):
    '''
    A wrapper to use the code in downloadSoundings. Used here just to make things more consistent
    :param start_dt:
    :param end_dt:
    :param bbox:
    :param settings:
    :return: Pandas dataframe of sounding data (if it exists locally)
    '''

    data = downloadSoundings.main(start_dt, end_dt, bbox, settings, download=False)['data']

    return data


def satellite_olr(start, end, settings, bbox=None, timeclip=True, aggregate=False, timeseries=False):
    '''
    Gets satellite obs from local data store
    :param start: datetime.
    :param end: datetime.
    :param settings: configuration settings
    :param bbox: list of [xmin, ymin, xmax, ymax]
    :param timeclip: boolean. Clips the satellite data to the precise start and end times
    :param aggregate: boolean. If True, creates a time average, if False and timeseries True, returns a time series, but if timeseries False, returns the last image in the time series (i.e. the image closest to the end datetime)
    :param timeseries: boolean. If True, returns a timeseries, if False, depends on value of aggregate
    :return: satellite OLR data
    '''
    from extractSatellite import getAutoSatDetails

    dir_path = settings['datadir'].rstrip('/') + '/satellite_olr/' + settings['region_name'] + '/' + settings['location_name'] + '/'

    # Get the productid from the bbox
    satellite, area, area_name, productid, proj4string, img_bnd_coords = getAutoSatDetails(settings['bbox'])

    file_list = []
    this_date = start
    # Add and hour on the end datetime to cover the fact that on 00Z, it doesn't pick up the last time slice
    while this_date <= (end + dt.timedelta(days=1)):
        file = dir_path + this_date.strftime('%Y%m') + '/' + productid + '_' + this_date.strftime('%Y%m%d') + '.nc'
        if os.path.isfile(file):
            file_list.append(file)
        this_date += dt.timedelta(days=1)

    cubes = iris.load(file_list)
    cube = cubes.concatenate_cube()

    if timeclip:
        cube = sf.periodConstraint(cube, start, end, greedy=True)

    if aggregate:
        cube = cube.collapsed('time', iris.analysis.MEAN)
    else:
        if not timeseries:
            cube = cube[-1,...]

    return cube


def unified_model(start, end, settings, bbox=None, region_type='event', model_id=['all'], var='all', checkftp=False, fclt_clip=None, timeclip=False, aggregate=True, totals=True):
    '''
    Loads *already downloaded* UM data for the specified period, model_id, variables and subsets by bbox
    :param start: datetime object
    :param end: datetime object
    :param settings: dictionary of settings
    :param bbox: Optional. list of bounding box coordinates [xmin, ymin, xmax, ymax]
    :param region_type: String. Either 'all', 'event' (smallest box), 'region', or 'tropics' (largest box)
    :param model_id: Either not specified (i.e. 'all') or a string or a list of strings that matches ['analysis',
            'ga7', 'km4p4', 'km1p5']
    :param var: Either not specified (i.e. 'all') or a string or a list of strings that matches names in
            sf.get_default_stash_proc_codes()['name']
    :param checkftp: boolean. If True, the script will check on the ftp site for files not currently in the local
            filelist
    :param fclt_clip: tuple. If declared, this gives a forecast lead time range to clip to. E.g. (12, 24)
    :param timeclip: boolean. If True, uses the start and end datetimes to subset the model data by time.
            If False, it returns the full cube
    :param aggregate: boolean. Return a collapsed cube aggregated between start and end (True), or return all available timesteps within the period (False)
    :param totals: boolean. If aggregate=True, returns the total over the aggregation period (True), or the mean (False)
    :return: Cubelist of all variables and init_times
    '''
    full_file_list = get_local_flist(start, end, settings, region_type=region_type)
    file_vars = list(set([os.path.basename(fn).split('_')[2] for fn in full_file_list]))
    file_model_ids = list(set([os.path.basename(fn).split('_')[1] for fn in full_file_list]))

    if isinstance(var, str):
        var = [var]

    if isinstance(model_id, str):
        model_id = [model_id]

    if var == ['all']:
        # Gets all available
        vars = file_vars
    else:
        # subset file_vars according to the list given
        vars = [fv for v in var for fv in file_vars if v in fv]

    if model_id == ['all']:
        # Get all available
        model_ids = file_model_ids
    else:
        model_ids = [fmi for m in model_id for fmi in file_model_ids if m in fmi]

    if bbox and not isinstance(bbox, dict):
        bbox = {'xmin': bbox[0], 'ymin': bbox[1], 'xmax': bbox[2], 'ymax': bbox[3]}

    cube_dict = {}

    for mod in model_ids:

        mod_dict = {}
        for v in vars:

            print('   Loading:', mod, v)
            these_files = [x for x in full_file_list if v in x and mod in x]

            try:
                cubes = iris.load(these_files)
            except:
                print('Failed to load files into a cubelist')
                continue

            # Check cubes have loaded correctly
            attempt_reload = False
            for cube in cubes:
                if not cube.coords():
                    attempt_reload = True
            if attempt_reload:
                cubes = iris.load(these_files)

            ocubes = iris.cube.CubeList([])
            for cube in cubes:

                # Firstly, try to convert units
                try:
                    if cube.units == cf_units.Unit('kg m-2 s-1'):
                        cube.convert_units('kg m-2 h-1')
                except:
                    print("Can\'t change units")

                if bbox and (not region_type == 'region'):
                    # Clip the model output to a bounding box
                    try:
                        cube = cube.intersection(latitude=(bbox['ymin'], bbox['ymax']), longitude=(bbox['xmin'], bbox['xmax']))
                        cube.attributes['bboxclipped'] = 'True'
                    except:
                        continue
                else:
                    cube.attributes['bboxclipped'] = 'False'

                if fclt_clip:
                    ltcon = iris.Constraint(forecast_period= lambda fp: fclt_clip[0] <= fp < fclt_clip[1])
                    cube = cube.extract(ltcon)

                if timeclip:
                    # Clip the model output to the given time bounds
                    try:
                        cube = sf.periodConstraint(cube, start, end)
                        cube.attributes['timeclipped'] = 'True'
                    except:
                        continue
                    #  ... but only if the data is fully within the period between start datetime and end datetime
                    try:
                        timechk = sf.check_time_fully_within(cube, start=start, end=end)
                    except:
                        continue

                else:
                    cube.attributes['timeclipped'] = 'False'

                if aggregate:
                    # Aggregate the model output for the time bounds given
                    try:
                        cube = cube.collapsed('time', iris.analysis.MEAN)
                        cube.attributes['aggregated'] = 'True'
                    except:
                        # print('Aggregate failed:', mod)
                        cube.attributes['aggregated'] = 'False'
                        pass
                    if totals:
                        diff_hrs = (end - start).total_seconds() // 3600
                        cube.data = cube.data * diff_hrs
                        cube.attributes['units_note'] = 'Values represent total accumulated over the aggregation period'
                    else:
                        cube.attributes['units_note'] = 'Values represent mean rate over the aggregation period'
                else:
                    cube.attributes['aggregated'] = 'False'

                cube.attributes['title'] = mod + '_' + v

                try:
                    if not cube.coord('latitude').has_bounds():
                        cube.coord('latitude').guess_bounds()

                    if not cube.coord('longitude').has_bounds():
                        cube.coord('longitude').guess_bounds()
                except:
                    continue

                ocubes.append(cube)

            if not mod == 'analysis':
                mod_dict[v] = ocubes
            else:
                # With analysis data, we don't have a forecast lead time dimension, so we can concatenate cubes together nicely into a cube
                try:
                    equalise_attributes(ocubes)
                except:
                    # If the variable doesn't exist ocubes will be empty and return an error, so just ignore this
                    continue
                ocube = ocubes.concatenate_cube()
                if aggregate:
                    # Aggregate the analysis data for the time bounds given
                    try:
                        # pdb.set_trace()
                        ocube = ocube.collapsed('time', iris.analysis.MEAN)
                        ocube.attributes['aggregated'] = 'True'
                    except:
                        # print('Aggregate failed:', mod)
                        ocube.attributes['aggregated'] = 'False'
                        pass
                    if totals:
                        diff_hrs = (end - start).total_seconds() // 3600
                        ocube.data = ocube.data * diff_hrs
                        ocube.attributes['units_note'] = 'Values represent total accumulated over the aggregation period'
                    else:
                        ocube.attributes['units_note'] = 'Values represent mean rate over the aggregation period'
                else:
                    ocube.attributes['aggregated'] = 'False'

                ocube.attributes['title'] = mod + '_' + v

                # Set Bounds
                if not ocube.coord('latitude').has_bounds():
                    ocube.coord('latitude').bounds = None
                    ocube.coord('latitude').guess_bounds(1.0)
                if not ocube.coord('longitude').has_bounds():
                    ocube.coord('longitude').bounds = None
                    ocube.coord('longitude').guess_bounds(1.0)

                mod_dict[v] = ocube

        cube_dict[mod] = mod_dict

    return cube_dict


def gpm_imerg(start, end, settings, latency=None, bbox=None, quality=False, aggregate=False):
    '''
    Loads *already downloaded* GPM IMERG data (30-minute precipitation from satellite) and returns a cube
    :param start: datetime object
    :param end: datetime object
    :param settings: settings from the config file
    :param latency: Choose from 'NRTearly', 'NRTlate', 'production' or None (choose the longest latency)
    :param bbox: Optional. List of bounding box coordinates [xmin, ymin, xmax, ymax], or a dictionary with keys=[xmin, ymin, xmax, ymax]. If set, the cube returned will be clipped to these coordinates.
    :param quality: boolean (optional). Do we want to return the quality flag (True) or the actual data (False)
    :param aggregate: boolean (optional). If True, the function will return an aggregated 2D array, collapsed over time. For precip, this will be a total of precipitation during the period, for quality index, it will be a mean.
    :return: cube
    '''

    if not latency:
        latency = downloadGPM.gpm_latency_decider(end)

    inpath = settings['gpm_path'] + 'netcdf/imerg/'+latency+'/'

    if start > end:
        raise ValueError('You provided a start_date that comes after the end_date.')

    file_search = inpath + '%Y/gpm_imerg_'+latency+'_*_%Y%m%d.nc' if not quality else inpath + '%Y/gpm_imerg_'+latency+'_*_%Y%m%d_quality.nc'
    numdays = (end - start).days + 1
    file_list_wildcard = [(start + dt.timedelta(days=x)).strftime(file_search) for x in range(0, numdays)]
    file_list = []
    for fn in file_list_wildcard:
        file_part = glob.glob(fn.replace('.nc', '_part.nc'))
        file_full = glob.glob(fn)
        if len(file_full) == 1:
            file_list.extend(file_full)
        else:
            file_list.extend(file_part)

    newcubelist = []
    for file in file_list:
        if os.path.isfile(file):
            cube = iris.load_cube(file)
            # Make sure the order of the dims is correct
            mycoords = [c.name() for c in cube.coords()]
            if mycoords != ['time','latitude','longitude']:
                cube.transpose([mycoords.index('time'), mycoords.index('latitude'), mycoords.index('longitude')])
            newcubelist.append(cube)
        else:
            print('File (or part file) doesn\'t exist: ' + file)

    # Get Most common version
    def most_common(lst):
        return max(set(lst), key=lst.count)

    version = most_common([os.path.basename(fn).split('_')[3] for fn in file_list])

    newcubelist = iris.cube.CubeList(newcubelist)
    try:
        cube = newcubelist.concatenate_cube()
    except:
        print('concatenate_cube failed')
        cube = newcubelist.concatenate()[0]

    # Clip the cube to start and end
    cube = sf.periodConstraint(cube, start, end)

    if bbox:
        cube = sf.domainClip(cube, bbox)
    elif settings['bbox']:
        cube = sf.domainClip(cube, settings['bbox'])
    else:
        print('   Loading Global GPM data')

    if aggregate:
        if quality:
            cube = cube.collapsed('time', iris.analysis.MEAN)
        else:
            cube = cube.collapsed('time', iris.analysis.SUM)
            cube.data = cube.data / 2.
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()

    # Mask very low or zero values
    cube.data = ma.masked_less(cube.data, 0.6)
    # Set some metadata
    cube.attributes['STASH'] = iris.fileformats.pp.STASH(1, 5, 216)
    cube.attributes['data_source'] = 'GPM'
    cube.attributes['product_name'] = 'imerg' if not quality else 'imerg quality flag'
    cube.attributes['latency'] = latency
    cube.attributes['version'] = version
    cube.attributes['aggregated'] = str(aggregate)
    cube.attributes['title'] = 'GPM '+latency if not quality else 'GPM '+latency+ ' Quality Flag'

    return cube

