import glob
import os
from datetime import timedelta
import iris
import location_config as config
from iris.experimental.equalise_cubes import equalise_attributes
import std_functions as sf
from downloadUM import get_local_flist
import downloadSoundings
import downloadGPM

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


def unified_model(start, end, event_name, settings, bbox=None, region_type='event', model_id='all', var='all', checkftp=False, timeclip=False):
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
    # print(file_vars)

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
    # print(vars)

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
            # print(these_files)

            cubes = iris.load(these_files)
            ocubes = iris.cube.CubeList([])
            for cube in cubes:
                if bbox:
                    cube = cube.intersection(latitude=(bbox['ymin'], bbox['ymax']), longitude=(bbox['xmin'], bbox['xmax']))
                if timeclip:
                    cube = sf.periodConstraint(cube, start, end)
                ocubes.append(cube)

            try:
                equalise_attributes(ocubes)
                ocube = ocubes.concatenate_cube()
                mod_dict[var] = ocube
            except:
                mod_dict[var] = ocubes

        cube_dict[model_id] = mod_dict

    return cube_dict


def gpm_imerg(start, end, settings, latency=None, bbox=None, quality=False):
    '''
    Loads *already downloaded* GPM IMERG data (30-minute precipitation from satellite) and returns a cube
    :param start: datetime object
    :param end: datetime object
    :param settings: settings from the config file
    :param latency: Choose from 'NRTearly', 'NRTlate', or 'production'
    :param bbox: Optional. List of bounding box coordinates [xmin, ymin, xmax, ymax], or a dictionary with keys=[xmin, ymin, xmax, ymax]. If set, the cube returned will be clipped to these coordinates.
    :param quality: boolean (optional). Do we want to return the quality flag (True) or the actual data (False)
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

    if bbox:
        cube = sf.domainClip(cube, bbox)

    cube.attributes['STASH'] = iris.fileformats.pp.STASH(1, 5, 216)
    cube.attributes['data_source'] = 'GPM'
    cube.attributes['product_name'] = 'imerg' if not quality else 'imerg quality flag'
    cube.attributes['latency'] = latency
    cube.attributes['version'] = version

    return cube

