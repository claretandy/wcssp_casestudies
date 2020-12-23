import glob
import os
from datetime import timedelta
import iris
import cf_units
import location_config as config
from iris.experimental.equalise_cubes import equalise_attributes
import std_functions as sf
from downloadUM import get_local_flist
import downloadSoundings
import downloadGPM
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


def unified_model(start, end, event_name, settings, bbox=None, region_type='event', model_id='all', var='all', checkftp=False, timeclip=False, aggregate=True, totals=True):
    '''
    Loads *already downloaded* UM data for the specified period, model_id, variables and subsets by bbox
    :param start: datetime object
    :param end: datetime object
    :param event_name: string. Format is <region_name>/<date>_<event_name> (e.g. 'Java/20190101_ColdSurge') or 'RealTime'
    :param settings: settings from the config file
    :param bbox: Optional. list of bounding box coordinates [xmin, ymin, xmax, ymax]
    :param region_type: String. Either 'all', 'event' (smallest box), 'region', or 'tropics' (largest box)
    :param model_id: Either not specified (i.e. 'all') or a string or a list of strings that matches ['analysis',
            'ga7', 'km4p4', 'km1p5']
    :param var: Either not specified (i.e. 'all') or a string or a list of strings that matches names in
            sf.get_default_stash_proc_codes()['name']
    :param checkftp: boolean. If True, the script will check on the ftp site for files not currently in the local
            filelist
    :param timeclip: boolean. If True, uses the start and end datetimes to subset the model data by time.
            If False, it returns the full cube
    :return: Cubelist of all variables and init_times
    '''

    full_file_list = get_local_flist(start, end, event_name, settings, region_type=region_type)
    file_vars = list(set([os.path.basename(fn).split('_')[-2] for fn in full_file_list]))
    file_model_ids = list(set([os.path.basename(fn).split('_')[-3] for fn in full_file_list]))

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

    for model_id in model_ids:

        mod_dict = {}
        for var in vars:

            print('   Loading:', model_id, var)
            these_files = [x for x in full_file_list if var in x and model_id in x]
            try:
                cubes = iris.load(these_files)
            except:
                print('Failed to load files into a cubelist')
                continue

            ocubes = iris.cube.CubeList([])
            for cube in cubes:

                # Firstly, try to convert units
                try:
                    if cube.units == cf_units.Unit('kg m-2 s-1'):
                        cube.convert_units('kg m-2 h-1')
                except:
                    print("Can\'t change units")

                if bbox:
                    # Clip the model output to a bounding box
                    try:
                        cube = cube.intersection(latitude=(bbox['ymin'], bbox['ymax']), longitude=(bbox['xmin'], bbox['xmax']))
                        cube.attributes['bboxclipped'] = 'True'
                    except:
                        continue
                else:
                    cube.attributes['bboxclipped'] = 'False'

                if timeclip:
                    # Clip the model output to the given time bounds
                    #  ... but only if the data is fully within the period between start datetime and end datetime
                    timechk = sf.check_time_fully_within(cube, start=start, end=end)
                    if not timechk:
                        continue
                    try:
                        cube = sf.periodConstraint(cube, start, end)
                        cube.attributes['timeclipped'] = 'True'
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
                        print('Aggregate failed:', model_id)
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

                cube.attributes['title'] = model_id + '_' + var
                ocubes.append(cube)

            if model_id != 'analysis':
                mod_dict[var] = ocubes
            else:
                # With analysis data, we don't have a forecast lead time dimension, so we can concatenate cubes together nicely into a cube
                equalise_attributes(ocubes)
                ocube = ocubes.concatenate_cube()
                if aggregate:
                    # Aggregate the analysis data for the time bounds given
                    try:
                        ocube = ocube.collapsed('time', iris.analysis.MEAN)
                        ocube.attributes['aggregated'] = 'True'
                    except:
                        print('Aggregate failed:', model_id)
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
                ocube.attributes['title'] = model_id + '_' + var
                mod_dict[var] = ocube

        cube_dict[model_id] = mod_dict

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
    file_list_wildcard = [(start + timedelta(days=x)).strftime(file_search) for x in range(0, numdays)]
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

    if aggregate:
        if quality:
            cube = cube.collapsed('time', iris.analysis.MEAN)
        else:
            cube = cube.collapsed('time', iris.analysis.SUM)
            cube.data = cube.data / 2.
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()

    cube.attributes['STASH'] = iris.fileformats.pp.STASH(1, 5, 216)
    cube.attributes['data_source'] = 'GPM'
    cube.attributes['product_name'] = 'imerg' if not quality else 'imerg quality flag'
    cube.attributes['latency'] = latency
    cube.attributes['version'] = version
    cube.attributes['aggregated'] = str(aggregate)
    cube.attributes['title'] = 'GPM '+latency if not quality else 'GPM '+latency+ ' Quality Flag'

    return cube

