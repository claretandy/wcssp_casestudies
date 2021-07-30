import datetime as dt
import os, sys
import iris
from iris.coord_categorisation import add_categorised_coord
import subprocess
import numpy as np
import numpy.ma as ma
import pathlib
import cf_units
import glob
from osgeo import gdal
from shapely.geometry import Polygon
from collections import Counter
from dateutil.relativedelta import relativedelta
import pdb

def getModelDomain_bybox(bbox):

    xmin, ymin, xmax, ymax = bbox
    p1 = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    seasia  = Polygon([(90, -18), (90, 30), (154, 30), (154, -18)])
    tafrica = Polygon([(-19, -12), (-19, 22), (52, 22), (52, -12)])

    if p1.intersects(seasia):
        domain = 'SEAsia'
    elif p1.intersects(tafrica):
        domain = 'TAfrica'
    else:
        print('The bbox does not match any available convective scale models')
        domain = 'Global'

    return domain


def getOneModelName(model_ids, modtype='global'):
    '''
    For use in plotting. We often need to pick one model id for plotting analysis vs global vs regional. This function attempts to choose from a list of full model ids, using an ordered list of keywords.
    For example, we might have a model_id = ['SEA3_n1280_ga6', 'SEA3_n1280_ga7plus', 'SEA3_km4p4_protora1t']. The problem here is that we have two global model ids ('SEA3_n1280_ga6' and 'SEA3_n1280_ga7plus'), but no way of choosing which one to use. The order of the list in this function 'poss_glo_keywords' would mean that we select 'SEA3_n1280_ga6'.
    :param model_ids: list of available model ids (as they occur in the filename)
    :return: one global and one regional model id to use
    '''

    import itertools

    # The order is important here, because the first one to occur will be selected
    if modtype == 'global':
        poss_keywords = ['ga6', 'ga7', 'global']
    elif modtype == 'regional':
        poss_keywords = ['km4p4', 'km1p5', 'africa']
    else:
        print('modtype needs to be either \'global\' or \'regional\'')
        return

    outmod = ''
    for pg, mod in itertools.product(poss_keywords, model_ids):
        if pg in mod:
            outmod = mod
            break

    return outmod


def getBBox_byRegionName(regname):

    regname = regname.lower()
    known_regions = {
        'seasia' : [90, -18, 154, 30],
        'tafrica': [-19, -12, 52, 22],
        'global' : [-180, -90, 180, 90]
    }

    try:
        return known_regions[regname]
    except:
        return 'Region unknown: ' + regname


def getRegionBBox_byBBox(bbox):

    xmin, ymin, xmax, ymax = bbox
    p1 = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymin)])

    outdict = {}

    reg_dict = {
        'East-Africa': Polygon([(28, -12), (28, 6), (48, 6), (48, -12)]),
        'West-Africa': Polygon([(-18, -2), (-18, 21), (25, 21), (25, -2)]),
        'SE-Asia': Polygon([(90, -18), (90, 30), (154, 30), (154, -18)])
    }

    for k, v in reg_dict.items():
        if p1.intersects(v):
            outdict['region_name'] = k
            outdict['region_bbox'] = list(v.bounds)

    return outdict

def getModels_bybox(bbox, reg=None):

    xmin, ymin, xmax, ymax = bbox
    p1 = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymin)])

    if not reg:
        reg = getModelDomain_bybox(bbox)

    domain_dict = {
        'SEAsia' : {
            'ga6'     : Polygon([(90, -18), (90, 30), (154, 30), (154, -18)]),
            'ga7'     : Polygon([(90, -18), (90, 30), (154, 30), (154, -18)]),
            'km4p4'   : Polygon([(90, -18), (90, 30), (154, 30), (154, -18)]),
            'phikm1p5': Polygon([(116, 3), (116, 21), (132, 21), (132, 3)]),
            'viekm1p5': Polygon([(96.26825, 4.23075), (96.26825, 25.155752), (111.11825, 25.155752), (111.11825, 4.23075)]),
            'malkm1p5': Polygon([(95.5, -2.5), (95.5, 10.5), (108.5, 10.5), (108.5, -2.5)]),
            'indkm1p5': Polygon([(100, -15), (100, 1), (120, 1), (120, -15)]),
            'analysis': Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)])
        },
        'TAfrica' : {
            'ga6'      : Polygon([(-19, -12), (-19, 22), (52, 22), (52, -12)]),
            'ga7'      : Polygon([(-19, -12), (-19, 22), (52, 22), (52, -12)]),
            'takm4p4'  : Polygon([(-19, -12), (-19, 22), (52, 22), (52, -12)]),
            'eakm4p4'  : Polygon([(21.49, -20.52), (21.49, 17.48), (52, 17.48), (52, -20.52)]),
            'africa-prods' : Polygon([(-19, -12), (-19, 22), (52, 22), (52, -12)]),
            'global-prods' : Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)]),
            'analysis': Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)])
        },
        'Global': {
            'global-prods': Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)]),
            'analysis': Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)])
        }
    }
    model_list = []
    domain_list = []
    area_list = []
    for k, v in domain_dict.items():
        for k1, v1 in v.items():
            if p1.intersects(v1) and k == reg:
                model_list.append(k1)
                domain_list.append(k)
                area_list.append(p1.intersection(v1).area)

    # Accounts for the fact that the global model matches everything
    domain = domain_list[area_list.index(np.max(area_list))]  # Matches the largest area
    # bbox = Counter(domain_list).most_common()[0][0]

    if not domain or model_list == []:
        print('The bbox does not match any available convective scale model domains')
        pdb.set_trace()

    return {"bbox" : domain, "model_list": model_list}


def getJobID_byDateTime(thisdate, domain='SEAsia', choice='newest', searchtxt=None):
    # Preferred method here ...
    # Provide a date, and return either the newest OR the oldest running model_id
    # NB: Update this when new model versions are run

    if searchtxt:
        if isinstance(searchtxt, list):
            searchtxt = searchtxt[0]
        if searchtxt == 'analysis':
            domain = 'Global'

    if domain == 'SEAsia':
        dtrng_byjobid = {'u-ai613': [dt.datetime(2016, 12, 15, 12, 0), dt.datetime(2017, 7, 17, 12, 0)],
                         'u-ao347': [dt.datetime(2017, 7, 13, 12, 0), dt.datetime(2017, 8, 11, 0, 0)],
                         'u-ao902': [dt.datetime(2017, 7, 28, 12, 0), dt.datetime(2019, 1, 15, 0, 0)],
                         'u-ba482': [dt.datetime(2018, 9, 6, 12, 0), dt.datetime(2019, 10, 30, 0, 0)], # started earier, but the first date that ra1tld appears
                         'u-bn272': [dt.datetime(2019, 10, 2, 0, 0), dt.datetime(2020, 8, 16, 0, 0)],
                         'u-bw324': [dt.datetime(2020, 8, 15, 0, 0), dt.datetime.now()]
                         }
    elif domain == 'TAfrica':
        dtrng_byjobid = {'u-ao907': [dt.datetime(2017, 7, 30, 12, 0), dt.datetime(2018, 10, 9, 0)],
                         'africa-psuite42': [dt.datetime(2018, 10, 9, 6, 0), dt.datetime(2019, 3, 11, 18)],
                         'africa-opfc': [dt.datetime(2019, 3, 12, 6), dt.datetime.now()]
                         }
    elif domain == 'Global':
        dtrng_byjobid = {'global-opfc': [dt.datetime(2015, 1, 1, 0), dt.datetime.now()]}
    else:
        print('Domain not specified')
        return

    allavail_jobid = []
    for jobid in dtrng_byjobid.keys():
        start, end = dtrng_byjobid[jobid]
        if start < thisdate < end:
            allavail_jobid.append(jobid)

    if choice == 'newest':
        try:
            outjobid = allavail_jobid[-1]
        except:
            outjobid = None
    elif choice == 'most_common':
        outjobid = most_common(allavail_jobid)
    elif choice == 'first':
        outjobid = allavail_jobid[0]
    else:
        outjobid = allavail_jobid[-1]

    return outjobid


def getModelID_byJobID(jobid, searchtxt=None):
    # Prefered method here ...
    modellist = {'u-ai613': ['SEA1_n1280_ga6', 'SEA1_n1280_ga7plus', 'SEA1_km4p4_sing4p0', 'PHI_km1p5_vn4p0', 'MAL_km1p5_vn4p0',
                                'INDON_km1p5_singv4p0', 'SINGV3_km1p5_singv4p0'],
                 'u-ao347': ['SEA2_n1280_ga6', 'SEA2_n1280_ga7plus', 'SEA2_km4p4_protora1t', 'PHI2_km1p5_protora1t',
                                'MAL2_km1p5_protora1t', 'INDON2_km1p5_protora1t', 'SINGRA1_km1p54p1_v4p1', 'SINGRA1_km1p5ra1'],
                 'u-ao902': ['SEA3_n1280_ga6', 'SEA3_n1280_ga7plus', 'SEA3_km4p4_protora1t', 'SEA3_phi2km1p5_protora1t',
                                'SEA3_mal2km1p5_protora1t', 'SEA3_indon2km1p5_protora1t', 'SINGRA1_km1p5ra1_protora1t', 'SINGRA1_km1p5v4p1_v4p1'],
                 'u-ao907': ['TAFRICA2_takm4p4_protora1t', 'TAFRICA2_n1280_ga7plus', 'TAFRICA2_n1280_ga6', 'TAFRICA2_eakm4p4_eur'],
                 'u-ba482': ['SEA4_n1280_ga6', 'SEA4_km4p4_ra1tld', 'SEA4_indkm1p5_ra1tld', 'SEA4_indkm0p2_ra1tld', 'SEA4_indkm0p5_ra1tld',
                                'SEA4_malkm1p5_ra1tld', 'SEA4_malkm0p2_ra1tld', 'SEA4_malkm0p5_ra1tld', 'SEA4_phikm1p5_ra1tld', 'SEA4_phikm0p2_ra1tld',
                                'SEA4_phikm0p5_ra1tld', 'SEA4_viekm1p5_ra1tld', 'analysis'], # NB: not all these are available from August 2018
                 'u-bn272': ['SEA5_n1280_ga7', 'SEA5_km4p4_ra2t', 'SEA5_indkm1p5_ra2t', 'SEA5_malkm1p5_ra2t', 'SEA5_phikm1p5_ra2t', 'SEA5_viekm1p5_ra2t', 'analysis'],
                 'u-bw324': ['SEA5_n1280_ga7', 'SEA5_km4p4_ra2t', 'SEA5_indkm1p5_ra2t', 'SEA5_malkm1p5_ra2t',
                                'SEA5_phikm1p5_ra2t', 'SEA5_viekm1p5_ra2t', 'analysis'],
                 'africa-opfc': ['africa-prods', 'global-prods', 'analysis'],
                 'africa-psuite42': ['africa-prods', 'global-prods', 'analysis'],
                 'global-opfc': ['global-prods', 'analysis']
        }

    if not searchtxt:
        outmodellist = modellist[jobid]
    elif isinstance(searchtxt, list):
        outmodellist = []
        for m in modellist[jobid]:
            for s in searchtxt:
                if s in m:
                    outmodellist.append(m)
    else:
        outmodellist = [m for m in modellist[jobid] if searchtxt in m]

    if outmodellist == []:
        return None
    else:
        return outmodellist

    
def getModelID_byDatetime_and_bbox(thisdate, bbox, searchtxt=False):

    domain = getModelDomain_bybox(bbox)
    jobid = getJobID_byDateTime(thisdate, domain=domain, choice='newest')

    if not searchtxt:
        searchtxt = getModels_bybox(bbox)['model_list']

    modellist = getModelID_byJobID(jobid, searchtxt=searchtxt)

    return {"jobid": jobid, "model_list": modellist}


def make_timeseries(start, end, freq):
    '''
    Makes a timeseries between the start and end with timestamps on common valid hours separated by freq
    e.g. 00, 03, 06 ... or 00, 01, 02, 03 ...
    :param start: datetime
    :param end: datetime
    :param freq: int (hours).
    :return: list of datetimes
    '''
    # Set the empty list to return
    outlist = []

    # Retrieve the end point
    hrs = np.arange(0,24,freq)
    for iday in [end - dt.timedelta(days=1), end]:
        for ih in hrs:
            pi = dt.datetime(iday.year, iday.month, iday.day, ih)
            if pi <= end:
                thisdt = pi

    # Looping backwards, get all the datetimes greater than or equal to the specified start
    while thisdt >= start:
        outlist.append(thisdt)
        thisdt -= dt.timedelta(hours=freq)

    # Return a sorted list
    return sorted(outlist)

def periodConstraint(cube, t1, t2, greedy=False):
    # Constrains the cube according to min and max datetimes
    def make_time_func(t1m, t2m, greedy=False):
        def tfunc(cell):
            if greedy:
                return t1m <= cell.point <= t2m
            else:
                return t1m < cell.point <= t2m
        return tfunc

    tfunc = make_time_func(t1, t2, greedy=greedy)
    tconst = iris.Constraint(time=tfunc)
    ocube = cube.extract(tconst)

    return(ocube)


def check_time_fully_within(cube, start=None, end=None, timeagg=None):
    """
    Checks whether the cube has start and end times fully within the specified start and end datetimes.
    If only timeagg is supplied, it will just check that the cube timespan is greater than timeagg
    :param cube: iris cube with a time coord
    :param start: datetime
    :param end: datetime
    :param timeagg: int
    :return: boolean
    """

    myu = cube.coord('time').units
    if not cube.coord('time').has_bounds():
        try:
            cube.coord('time').guess_bounds()
        except:
            return True

    bnd_start = min([myu.num2date(x[0]) for x in cube.coord('time').bounds])
    bnd_end = max([myu.num2date(x[1]) for x in cube.coord('time').bounds])
    diff_hrs = int( np.round((bnd_end - bnd_start).total_seconds() / 3600., 0) )

    if start and end:
        timeagg = int( np.round((end - start).total_seconds() / 3600., 0) )

    if diff_hrs >= timeagg:
        return True
    else:
        return False



def loadModelData(start, end, stash, plotdomain, settings, searchtxt=None, lbproc=0, aggregate=True, totals=True, overwrite=False):
    """
    Loads all available model runs and clips data:
        - spatially (within lat/on box specified by bbox) and
        - temporally (between start and end datetimes)
    :param start: datetime
    :param end: datetime
    :param stash: string (short format e.g. 4203)
    :param plotdomain: list (format [xmin, ymin, xmax, ymax] each value is a float)
    :param searchtxt: string (optional). Name of the model (e.g. km4p4, km1p5, ga6, ga7, africa-opfc, global-opfc, africa-psuite42 etc)
    :param lbproc: int (optional). Assumes we want 0 (instantaneous data), but 128, 4096 and 8192 are also possible
    :param aggregate: boolean. Aggregate over the start-end period or not?
    :param totals: boolean. Do the values represent the total accumulated over the aggregation period?
    :param overwrite: Overwrite files downloaded from MASS?
    :return: CubeList of all available model runs between the start and end
    """

    # 1. Get model bbox for the given plotting bbox
    domain = getModelDomain_bybox(plotdomain)
    if not domain:
        print("Not loading data because no data in the plotting bbox")
        return None

    jobid = getJobID_byDateTime(end, domain=domain)
    model_id = getModelID_byJobID(jobid, searchtxt=searchtxt)
    if not model_id:
        print("The searchtxt \'"+searchtxt+"\' does\'t exist in this jobid: "+jobid)
        return None
    model_id = model_id[0] if isinstance(model_id, list) else model_id

    # Directory that model data is saved to
    modeldatadir = settings['scratchdir'] + 'ModelData/'
    odir = pathlib.PurePath(modeldatadir, jobid).as_posix()
    if not pathlib.Path(odir).is_dir():
        pathlib.Path(odir).mkdir(parents=True)

    # 2. Extract from MASS
    # pdb.set_trace()
    if str(stash) in ['4201', '4203', '5216', '5226']:
        stash = getPrecipStash(model_id, type='short')

    print(start, end, model_id, jobid, stash, lbproc)
    # pdb.set_trace()
    cubelist = selectModelDataFromMASS(getInitTimes(start, end, domain, model_id=model_id), stash, odir=odir, domain=domain, plotdomain=plotdomain, lbproc=lbproc, choice='newest',searchtxt=searchtxt, returncube=True, overwrite=overwrite)
    # pdb.set_trace()
    # 3. Loop through data and load into a cube list
    outcubelist = iris.cube.CubeList([])
    for cube in cubelist:
        # 3A. Subset to plot bbox
        #print(cube)
        try:
            cube_dclipped = cube.intersection(latitude=(plotdomain[1],plotdomain[3]), longitude=(plotdomain[0],plotdomain[2]))
            # 3B. Extract time period
            cube_tclipped = periodConstraint(cube_dclipped, start, end)
            try:
                timechk = check_time_fully_within(cube_tclipped, start=start, end=end)
            except:
                continue

            if not timechk :
                continue

            # 3C. Convert units
            try:
                if cube_tclipped.units == cf_units.Unit('kg m-2 s-1'):
                    cube_tclipped.convert_units('kg m-2 h-1')
            except:
                print("Can\'t change units")


            # 3D. Aggregate
            try:
                if aggregate:
                    if len(cube_tclipped.coord('time').points) > 1:
                        cube_tclipped = cube_tclipped.collapsed('time', iris.analysis.MEAN) # Assuming precip
                        if totals:
                            diff_hrs = (end - start).total_seconds() // 3600
                            # print(diff_hrs)
                            cube_tclipped.data = cube_tclipped.data * diff_hrs
                            cube_tclipped.attributes['units_note'] = 'Values represent total accumulated over the aggregation period'
                        else:
                            cube_tclipped.attributes['units_note'] = 'Values represent mean rate over the aggregation period'
                        cube_tclipped.attributes['aggregated'] = 'True'
                else:
                    cube_tclipped.attributes['units_note'] = 'Values represent those given by cube.units'
            except:
                print('****************************************')
                print('There\'s a problem with this cube ....')
                print(cube)
                print('****************************************')
                # pdb.set_trace()

            # 3E. Add attributes
            try:
                cube_tclipped.attributes['title'] = model_id
            except AttributeError:
                print('Can\'t add attributes')
            outcubelist.append(cube_tclipped)
        except IndexError:
            print(model_id, 'not in bbox')

    return(outcubelist)


def createBuffer(input, bufferDist):
    '''
    :param input: Can be either a filename or an OGR dataset
    :param bufferDist: Distance to buffer in the coordinate units of the input dataset (e.g. for lat/lon data, a value of 1 = 112km at the equator)
    :return: An OGR dataset containing the buffer
    '''

    from osgeo import ogr
    import os

    # Testing ....
    # input = '/data/users/hadhy/GIS_Data/Global/test_coast3.shp'
    # bufferDist = 0.4

    if isinstance(input, str):
        inputds = ogr.Open(input)
        outshp = input.replace('.shp','_buffer.shp')
    elif isinstance(input, ogr.DataSource):
        inputds = input
        outshp = 'tmp_buffer.shp' # In the current working directory
    else:
        sys.exit('Didn\'t recognise the input to createBuffer')

    # inputds = ogr.Open(inshp)
    inputlyr = inputds.GetLayer()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outshp):
        shpdriver.DeleteDataSource(outshp)
    outputBufferds = shpdriver.CreateDataSource(outshp)
    bufferlyr = outputBufferds.CreateLayer(outshp, geom_type=ogr.wkbPolygon)
    bufferlyr.CreateField(ogr.FieldDefn("ID", ogr.OFTInteger))
    bufferlyr.CreateField(ogr.FieldDefn("Buffer", ogr.OFTInteger))
    featureDefn = bufferlyr.GetLayerDefn()

    feature = inputlyr.GetNextFeature()
    while feature:
        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(bufferDist)

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        outFeature.SetField("ID", feature.GetFID())
        outFeature.SetField("Buffer", 1)
        bufferlyr.CreateFeature(outFeature)

        outFeature = None
        feature = inputlyr.GetNextFeature()

    outputBufferds = None

    if isinstance(input, str):
        return outshp
    elif isinstance(input, ogr.DataSource):
        print('Created: ' + os.getcwd() + '/' + outshp)
        ds = ogr.Open(outshp)
        return ds
    else:
        return



def poly2cube(shpfile, attribute, cube, dtype=gdal.GDT_Float64):
    '''
    Returns a cube with values for the given attribute from the shapefile
    :param shpfile: An ESRI *.shp file
    :param attribute: An integer formatted column in the shapefile table
    :param cube: An iris cube on which the polygons will be mapped on to
    :return: A cube with values for the given attribute
    '''

    from osgeo import ogr
    from osgeo import osr
    from osgeo import gdal

    # 1. create a gdal dataset from the cube
    ds = cube2gdalds(cube, dtype=dtype)

    # 2. Rasterize the vectors
    vec = ogr.Open(shpfile)
    veclyr = vec.GetLayer()

    # if not veclyr.GetSpatialRef():
    #     print('Assume WGS84')
    #     osr.SpatialReference()

    # fieldnames = [field.name for field in veclyr.schema]
    # print(fieldnames)
    # print(ds.GetGeoTransform())
    # print(attribute)
    # print("\"ATTRIBUTE=" + attribute + "\"")

    if attribute == '':
        gdal.RasterizeLayer(ds, [1], veclyr, burn_values=[1])
    else:
        gdal.RasterizeLayer(ds, [1], veclyr, burn_values=[255]) # options=["\"attribute=" + attribute + "\""])

    ds.GetRasterBand(1).SetNoDataValue(0.0)
    # 3. Convert the resulting gdal dataset back to a cube
    ocube = gdalds2cube(ds)
    ds = None

    return ocube


def domainClip(cube, bbox):
    '''
    Clips a cube according to a bounding box
    :param cube: An iris cube
    :param domain: list containing xmin, ymin, xmax, ymax or dictionary defining each
    :return: iris cube containing the clipped bbox
    '''

    if isinstance(bbox, dict):
        xmin, ymin, xmax, ymax = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
        # lonce = iris.coords.CoordExtent('longitude', bbox['xmin'], bbox['xmax'])
        # latce = iris.coords.CoordExtent('latitude', bbox['ymin'], bbox['ymax'])
    else:
        xmin, ymin, xmax, ymax = bbox = [float(b) for b in bbox]
        # lonce = iris.coords.CoordExtent('longitude', xmin, xmax)
        # latce = iris.coords.CoordExtent('latitude', ymin, ymax)

    cube_cropped = cube.intersection(longitude=(xmin, xmax), latitude=(ymin, ymax))

    return cube_cropped


def cube2gdalds(cube, empty=False, dtype=gdal.GDT_Byte):
    '''
    :param cube: iris.cube.Cube A 2D cube with bounds
    :param empty: boolean. If True, returns an empty gdal dataset, if False, uses the cube values
    :return: A GDAL dataset
    '''

    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds()
    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds()

    x_pixel_width = np.round(np.mean([x[1] - x[0] for x in cube.coord('longitude').bounds]), 6)
    y_pixel_width = np.round(np.mean([x[1] - x[0] for x in cube.coord('latitude').bounds]), 6)
    y_pixel_width = y_pixel_width if y_pixel_width < 0 else -1 * y_pixel_width

    x_res = cube.coord('longitude').points.shape[0]
    y_res = cube.coord('latitude').points.shape[0]
    x_min = np.round(np.min([np.min(x) for x in cube.coord('longitude').bounds]), 5)
    y_max = np.round(np.max([np.max(y) for y in cube.coord('latitude').bounds]), 5)

    ds = gdal.GetDriverByName("MEM").Create("", x_res, y_res, 1, dtype)
    ds.SetGeoTransform((x_min, x_pixel_width, 0, y_max, 0, y_pixel_width))
    band = ds.GetRasterBand(1)
    if not empty:
        if isinstance(cube.data, ma.MaskedArray):
            data2write = ma.filled(cube.data)
            ndval = cube.data.fill_value
        else:
            data2write = cube.data
            ndval = 0
        band.WriteArray(data2write)
        band.SetNoDataValue(ndval)

    # band.FlushCache()

    return ds


def gdalds2cube(gdalds, timestamp=None):

    '''
    :param gdalds: Input can be either a geotiff file or a gdal dataset object
    :param timestamp: Datetime (as a string formatted %Y%m%d%H%M)
    :return: iris cube
    '''

    from osgeo import gdal

    if isinstance(gdalds, str):
        ds = gdal.Open(gdalds)
    elif isinstance(gdalds, gdal.Dataset):
        ds = gdalds
    else:
        sys.exit('Didn\'t recognise the input to gdalds2cube')

    ulX, xDist, rtnX, ulY, rtnY, yDist = ds.GetGeoTransform()
    xDist = np.round(xDist, 6) # Round because sometimes gdal spits out too precise coordinates
    yDist = np.round(yDist, 6)
    yDist = yDist if yDist < 0.0 else yDist * -1 # The y cell length should be negative because we use the UPPER Y to calculate LOWER Y
    ncellx = ds.RasterXSize
    ncelly = ds.RasterYSize
    lrX    = (ulX + (ncellx * xDist)) - (xDist * 0.5)
    lrY    = (ulY + (ncelly * yDist)) - (yDist * 0.5)
    ulX    = ulX + ( xDist * 0.5 ) # Assuming the GT coord refers to the ul of the pixel
    ulY    = ulY + ( yDist * 0.5) # Assuming the GT coord refers to the ul of the pixel
    # proj   = ds.GetProjection()

    latcoord = iris.coords.DimCoord(np.arange(ulY, lrY + yDist, yDist), standard_name='latitude', units=cf_units.Unit('degrees'), coord_system=iris.coord_systems.GeogCS(6371229.0))
    loncoord = iris.coords.DimCoord(np.arange(ulX, lrX + xDist, xDist), standard_name='longitude', units=cf_units.Unit('degrees'), coord_system=iris.coord_systems.GeogCS(6371229.0))
    latcoord.guess_bounds()
    loncoord.guess_bounds()

    if ds.GetLayerCount() == 0:
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray()
    else:
        print('More than 1 raster band. What do we do next?')
        pdb.set_trace()

    nodatavalue = band.GetNoDataValue()
    if not nodatavalue == None:
        array = ma.masked_equal(array, nodatavalue)

    if timestamp:
        u = cf_units.Unit('hours since 1970-01-01 00:00:00', calendar=cf_units.CALENDAR_STANDARD)
        timecoord = iris.coords.DimCoord(u.date2num(dt.datetime.strptime(timestamp, '%Y%m%d%H%M')), standard_name='time', units=u)
        array = array[np.newaxis, ...]
        cube = iris.cube.Cube(array, dim_coords_and_dims=[(timecoord, 0), (latcoord, 1), (loncoord, 2)])
    else:
        cube = iris.cube.Cube(array, dim_coords_and_dims=[(latcoord, 0), (loncoord, 1)])

    return cube


def makeGlobalCube(resolution=0.5, landmask=True, timestamp=None):
    '''
    Create a new cube at a given grid resolution. The new cube can optionally contain a time dimension and a landmask
    :param resolution: float. Grid cell spatial resolution, used for x and y dimension
    :param landmask: boolean. If True, the function will use the iris/natural earth global polygons to create a land mask
    :param timestamp: Either None or datetime object. If a datetime is given, a time dimension will be created
    :return: iris.cube.Cube
    '''

    loncoord = iris.coords.DimCoord(np.arange(-180, 180, resolution), standard_name='longitude', units=cf_units.Unit('degrees'), coord_system=iris.coord_systems.GeogCS(6371229.0))
    loncoord.guess_bounds(bound_position=0)
    latcoord = iris.coords.DimCoord(np.arange(90, -90, -resolution), standard_name='latitude', units=cf_units.Unit('degrees'), coord_system=iris.coord_systems.GeogCS(6371229.0))
    latcoord.guess_bounds(bound_position=0)

    array = np.zeros((len(latcoord.points), len(loncoord.points)))

    if timestamp:
        u = cf_units.Unit('hours since 1970-01-01 00:00:00', calendar=cf_units.CALENDAR_STANDARD)
        timecoord = iris.coords.DimCoord(u.date2num(dt.datetime.strptime(timestamp, '%Y%m%d%H%M')), standard_name='time', units=u)
        array = array[np.newaxis, ...]
        cube = iris.cube.Cube(array, dim_coords_and_dims=[(timecoord, 0), (latcoord, 1), (loncoord, 2)])

    else:
        cube = iris.cube.Cube(array, dim_coords_and_dims=[(latcoord, 0), (loncoord, 1)])

    if landmask:
        from cartopy.io import shapereader
        shp = shapereader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
        cube = poly2cube(shp, '', cube)

    return cube


def getCubeBBox(cube, outtype='polygon'):

    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds()

    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds()

    if cube.coord('latitude').bounds[0][0] > cube.coord('latitude').bounds[0][1]:
        i, j = [1, 0]
    else:
        i, j = [0, 1]
    ymin = np.min([bnd[i] for bnd in cube.coord('latitude').bounds])
    ymax = np.max([bnd[j] for bnd in cube.coord('latitude').bounds])

    if cube.coord('longitude').bounds[0][0] > cube.coord('longitude').bounds[0][1]:
        k, l = [0, 1]
    else:
        k, l = [1, 0]
    xmin = np.min([bnd[l] for bnd in cube.coord('longitude').bounds])
    xmax = np.max([bnd[k] for bnd in cube.coord('longitude').bounds])

    if outtype == 'list':
        return [xmin, ymin, xmax, ymax]
    elif outtype == 'polygon':
        return Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymin)])
    else:
        return [xmin, ymin, xmax, ymax]


def get_fc_length(jobid, model_id):
    '''
    :param model_id: This is the ID of the model within the job, as a string
    :return: The length of model run in hours
    '''

    if isinstance(model_id, list):
        model_id = model_id[0]

    if 'km1p5' in model_id:
        fcl = 48 # 2 days ... actually this changes depending on the jobid
    elif 'km4p4' in model_id or 'ga' in model_id:
        fcl = 120 # 5 days
    elif 'africa' in jobid:
        fcl = 54 # 2 and a bit days (Operational TAfrica)
    elif 'global' in jobid:
        fcl = 168 # Operational Global
    else:
        print('Guessing the forecast length as 120 hours')
        fcl = 120

    return fcl


def get_default_stash_proc_codes(list_type='long'):
    '''
    Returns a pandas dataframe of the
    :param list_type: 'short' or 'long' or 'profile'
    :return: pandas dataframe
    '''
    import pandas as pd

    if list_type in ['long', 'short', 'profile', 'share']:
        list_type = list_type + '_list'

    df = pd.read_csv('std_stashcodes.csv')
    outdf = df[ df[list_type] & (df.share_tropics | df.share_region | df.share_event) ]

    return outdf


def get_fc_InitHours(jobid):

    initdict = {
        'analysis' : [0, 6, 12, 18],
        'global-opfc': [0, 6, 12, 18],
        'africa-opfc' : [6,18],
        'africa-psuite42': [6, 18],
        'u-ao907'   : [0,12],
        'u-ai613'   : [0,12],
        'u-ao347'   : [0,12],
        'u-ao902'   : [0,12],
        'u-ba482'   : [0,12],
        'u-bn272'   : [0,12],
        'u-bw324'   : [0,12]
    }
    try:
        init_hrs = initdict[jobid]
    except:
        print('Guessing 0Z and 12Z start times')
        init_hrs = [0,12]

    return(init_hrs)


def get_fc_start(start, end, fclt_range, bbox):
    '''
    Finds the model initialisation time that best fits the plot time range, and forecast lead time (FCLT) range. Search starts from the earliest FCLT, and works forwards in time. For example, if we are plotting from 06UTC on 4th Feb 2018 to 06UTC on 5th Feb 2018, and we want to find a single model run in the approximate forecast lead time range of T+6 to T+18, then this function will return the model initialisation time of 00UTC 4th Feb. If the FCLT range was T+12 to T+24, it would return the initialisation time of 12UTC 3rd Feb.
    :param start:
    :param end:
    :param fclt_range:
    :param bbox:
    :return:
    '''

    import itertools

    model_domain = getModelDomain_bybox(bbox)
    possible_inits = getInitTimes(start, end, model_domain) # .reverse()
    possible_inits.reverse()
    fclt_start, fclt_end = fclt_range

    fc_start_dt = possible_inits[-1]
    fclts = np.arange(fclt_start, fclt_end + 1)

    for fclt, init in itertools.product(fclts, possible_inits):
        this_fclt = (start - init).total_seconds() / (60 * 60)
        # print(init, f'T+{this_fclt}', fclt)
        fc_start_dt = init
        if this_fclt >= fclt:
            break

    return fc_start_dt


def getInitTimes(start_dt, end_dt, domain, model_id=None, fcl=None, init_hrs=None, searchtxt=None, init_incr=None):
    '''
    Given a start and end date of a case study period, what are all the possible model runs that could be available?
    :param start_dt: datetime
    :param end_dt: datetime
    :param domain: string. Can be one of 'SEAsia', 'TAfrica' or 'Global'. This is used with the date to get the jobid
    :param model_id: string (optional). Name of the model within the jobid. See the function 'getModelID_byJobID', but generally
            can be one of 'ga6', 'ga7', 'km4p4', 'indkm1p5', 'malkm1p5', 'phikm1p5',
            or 'global-prods' (for the operational global)
    :param fcl: int (optional). Forecast length in hours. If not given, the function 'get_fc_length' will be used
    :param init_hrs: list (optional). What hours (in UTC) is the forecast initialised?
    :param searchtxt: string (optional). If model_id is not given, this will be used to search for the model_id
    :param init_incr: int (not used now). Legacy option that needs to be removed
    :return: list of datetimes
    '''

    jobid = getJobID_byDateTime(start_dt, domain=domain, choice='newest')

    if not model_id:
        model_id = getModelID_byJobID(jobid, searchtxt=searchtxt)

    if not fcl:
        # Guess the forecast length from the start date (assume it doesn't change)
        fcl = get_fc_length(jobid, model_id)

    if not init_hrs:
        init_hrs = get_fc_InitHours(jobid)

    # 1. Find the most recent init time BEFORE the end_dt
    for iday in [end_dt - dt.timedelta(days=1), end_dt]:
        for ih in init_hrs:
            pi = dt.datetime(iday.year, iday.month, iday.day, ih)
            if pi < end_dt:
                mostrecent_init = pi

    # 2. Find the datetime of oldest model run available ...
    for iday in [start_dt + dt.timedelta(days=1), start_dt]:
        for ih in sorted(init_hrs, reverse=True):
            pi = dt.datetime(iday.year, iday.month, iday.day, ih)
            if pi > start_dt:
                oldest_init = pi - dt.timedelta(hours=int(fcl))

    # print(start_dt, end_dt)
    # print(oldest_init, mostrecent_init)

    # 3. Loop through oldest to most recent init dates to create a timeseries
    init_ts = []

    while oldest_init <= mostrecent_init:
        if oldest_init.hour in init_hrs:
            init_ts.append(oldest_init) # .strftime('%Y%m%dT%H%MZ')
        oldest_init += dt.timedelta(hours=1)

    return(init_ts)


def most_common(lst):
    return max(set(lst), key=lst.count)


def myround(x, base=3):
    return int(base * np.floor(float(x)/base))


def myroundup(x, base=3):
    return int(base * np.ceil(float(x)/base))

def lut_lbproc(lbproc, type='short'):
    '''
    What does the lbproc code mean?
    :param lbproc: integer. Either 0, 128, 4096 or 8192
    :param type: string. Either 'short' or 'long'
    :return: string for use in a filename or plot
    '''

    lut = {
        0: {'short': 'inst', 'long': 'Instantaneous'},
        128: {'short': 'timeaver', 'long': 'Time Step Average'},
        4096: {'short': 'timemin', 'long': 'Time Step Minimum'},
        8192: {'short': 'timemax', 'long': 'Time Step Maximum'}
    }

    return lut[lbproc][type]

def lut_stash(sc, type='short'):
    '''
    Gets either the long or short description of the stash code
    :param sc: integer. Must relate to an item in the file std_stashcodes.csv
    :param style: string. Can be either 'short' or 'long'
    :return: either a string of the stash code name, or a list of matching stash codes
    '''

    stashcodes = get_default_stash_proc_codes(list_type='long')
    col = 'name' if type == 'short' else 'long_name'

    if isinstance(sc, str):
        sc = int(sc)

    outname = stashcodes.loc[stashcodes.stash == sc][col]

    return outname.to_string(index=False).lstrip(' ')

    # May want to consider adding more functionality in the future, as below
    # if sc == 'all':
    #     return {'stash': stashcodes['stash'].to_list() , 'name': stashcodes[col].to_list()}
    #     return dict([(stitems[0], val[style]) for val, stitems in zip(stashcodes.values(), stashcodes.items())])
    #
    # if isinstance(sc, int):
    #     return dict([(sc, stashcodes[sc][style])])
    #
    # if isinstance(sc, str):
    #     return dict([(stitems[0], val[style]) for val, stitems in zip(stashcodes.values(), stashcodes.items())
    #                  if (sc.lower() in val['long'].lower()) or
    #                     (sc.lower() in val['short'].lower()) or
    #                     (sc.lower() in val['alt1'].lower()) ])


def get_lbproc_by_stash(stash, jobid):
    # which lbproc codes go with which stash codes?
    #TODO Maybe better to write an 'if' statement to separate out global (section 5) from convection permitting
    lbproc128_avail = {'u-ai613': [23, 4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205],
                       'u-ao347': [23, 4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205],
                       'u-ao902': [23, 4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205],
                       'u-ao907': [23, 4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205],
                       'u-ba482': [4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205, 21100, 21104],
                       'u-bn272': [4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205, 21100, 21104],
                       'u-bw324': [4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205,
                                   21100, 21104],
                       'africa-opfc'   : [],
                       'global-opfc': [],
                       'africa-psuite42': [4201]
                       }

    lbproc4096_avail = {'u-ai613': [],
                        'u-ao347': [],
                        'u-ao902': [],
                        'u-ao907': [],
                        'u-ba482': [20080],
                        'u-bn272': [20080],
                        'u-bw324': [20080],
                        'africa-opfc': [],
                        'global-opfc': [],
                        'africa-psuite42': []
                        }

    lbproc8192_avail = {'u-ai613': [],
                        'u-ao347': [],
                        'u-ao902': [],
                        'u-ao907': [],
                        'u-ba482': [3463, 20080],
                        'u-bn272': [3463, 20080],
                        'u-bw324': [3463, 20080],
                        'africa-opfc': [],
                        'global-opfc': [],
                        'africa-psuite42': []
                        }

    # 4201, 4202, 5201, 5202, 5226, 23
    lbproc0_notavail = {'u-ai613': [],
                        'u-ao347': [],
                        'u-ao902': [],
                        'u-ao907': [4201, 4202, 5201, 5202, 5226, 23],
                        'u-ba482': [],
                        'u-bn272': [],
                        'u-bw324': [],
                        'africa-opfc': [],
                        'global-opfc': [],
                        'africa-psuite42': []
                        }

    oproc = []
    oproc.append(0) if int(stash) not in lbproc0_notavail[jobid] else oproc
    oproc.append(128) if int(stash) in lbproc128_avail[jobid] else oproc
    oproc.append(4096) if int(stash) in lbproc4096_avail[jobid] else oproc
    oproc.append(8192) if int(stash) in lbproc8192_avail[jobid] else oproc

    if not oproc:
        oproc = [0]

    return oproc


def plotNP(z, title='title'):
    import matplotlib.pyplot as plt
    plt.pcolormesh(z, cmap='RdBu', vmin=np.nanmin(z), vmax=np.nanmax(z))
    plt.title(title)
    plt.colorbar()
    plt.show()


def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars

def createPointShapefile(lats, lons, vals, fieldnames, outshp, returnogr=False):
    '''
    Creates a shapefile and OGR object from a list of lats, lons and values
    :param lons: list of floating point values that refer to longitude
    :param lats: list of floating point values that refer to latitude
    :param vals: 2D array of values that relate to each coordinate
    :param fieldnames: list of field names (same length as ncols in vals)
    :return: Either a filename where the point shapefile is saved or an OGR object
    '''

    import osgeo.ogr as ogr
    import osgeo.osr as osr
    import csv

    # TODO make this function generic so that it works more generally

    # This will fail, but include as an example ...
    # use a dictionary reader so we can access by field name
    reader = csv.DictReader(open("volcano_data.txt", "rb"),
                            delimiter='\t',
                            quoting=csv.QUOTE_NONE)

    # set up the shapefile driver
    driver = ogr.GetDriverByName("ESRI Shapefile")

    # create the data source
    data_source = driver.CreateDataSource(outshp)

    # create the spatial reference, WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # create the layer
    layer = data_source.CreateLayer("point_layer", srs, ogr.wkbPoint)

    # Add the fields we're interested in
    field_name = ogr.FieldDefn("Name", ogr.OFTString)
    field_name.SetWidth(24)
    layer.CreateField(field_name)
    field_region = ogr.FieldDefn("Region", ogr.OFTString)
    field_region.SetWidth(24)
    layer.CreateField(field_region)
    layer.CreateField(ogr.FieldDefn("Latitude", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("Longitude", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("Elevation", ogr.OFTInteger))

    # Process the text file and add the attributes and features to the shapefile
    for row in reader:
        # create the feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        feature.SetField("Name", row['Name'])
        feature.SetField("Region", row['Region'])
        feature.SetField("Latitude", row['Latitude'])
        feature.SetField("Longitude", row['Longitude'])
        feature.SetField("Elevation", row['Elev'])

        # create the WKT for the feature using Python string formatting
        wkt = "POINT(%f %f)" % (float(row['Longitude']), float(row['Latitude']))

        # Create the point from the Well Known Txt
        point = ogr.CreateGeometryFromWkt(wkt)

        # Set the feature geometry using the point
        feature.SetGeometry(point)
        # Create the feature in the layer (shapefile)
        layer.CreateFeature(feature)
        # Dereference the feature
        feature = None

    # Save and close the data source
    data_source = None

    return 'Finished'


def accumulated2sequential(accum_cubefile, returncube=False):

    '''
    Converts an accumulated field to a time sequential field.
    An accumulated field is a cube that accumulates a quantity (such as rainfall or lightning flashes) through the run. The bounds for each time step start at 0 (e.g. [0,3], [0,6], [0,9], ...)
    A sequential field is one which contains an accumulation for a quantity for only the last time step. The bounds for a given timestep start at the end of the last timestep. (e.g. [0,3], [3,6], [6,9] ... )
    :param accum_cubefile: a file containing accumulated quantities
    :param output_cube: boolean to select either cube output (True) or filename (False)
    :return: Filename of a sequential field
    '''

    print('Converting accumulation file to sequential ...')
    print('    ... ' + accum_cubefile)
    # Currently setup for 4201 to 4203 (note that lightning doesn't have a different stash for sequential data)
    stash_convert = [
        (iris.fileformats.pp.STASH(1, 4, 201), iris.fileformats.pp.STASH(1, 4, 203)),
        (iris.fileformats.pp.STASH(1, 21, 104), iris.fileformats.pp.STASH(1, 21, 104))
    ]

    cube = iris.load_cube(accum_cubefile)
    srcStash = cube.attributes['STASH']
    dstStash = [sconv[1] for sconv in stash_convert if sconv[0] == srcStash]
    dstStash = srcStash if dstStash == [] else dstStash[0]
    ofile = accum_cubefile.replace(str(srcStash[1]) + str(srcStash[2]), str(dstStash[1]) + str(dstStash[2]))

    if ofile == accum_cubefile:
        temp_accum_cubefile = accum_cubefile.replace('.nc', '_old.nc')
        os.rename(accum_cubefile, temp_accum_cubefile)
        cube = iris.load_cube(temp_accum_cubefile)

    try:
        processed = cube.attributes['Processed_accumulated']
    except:
        processed = False

    if processed:
        newcube = iris.load_cube(ofile)
    else:
        # Make time coord
        myu = cube.coord('time').units
        newtime = iris.coords.DimCoord([bnd[1] for bnd in cube.coord('time').bounds], standard_name='time', units=myu)
        newtime.guess_bounds(bound_position=1)

        # Make forecast_period coord
        fcp = cube.coord('forecast_period')
        newfcp = iris.coords.DimCoord([bnd[1] for bnd in fcp.bounds], standard_name=fcp.standard_name, var_name=fcp.var_name, units=fcp.units)
        newfcp.guess_bounds(bound_position=1)

        newdata = []
        slice0 = np.zeros((len(cube.coord('latitude').points), len(cube.coord('longitude').points)))
        newdata.append(slice0)

        # Make data
        for i, x_slice in enumerate(cube.slices(['latitude', 'longitude'])):
            if i == 0:
                lastslice = slice0
                continue
            slicedata = x_slice.data - lastslice
            newdata.append(slicedata)
            lastslice = x_slice.data

        newdatastk = np.stack(newdata, axis=0)

        newcube = cube.copy()
        newcube.data = newdatastk
        newcube.remove_coord('time')
        newcube.remove_coord('forecast_period')
        newcube.add_dim_coord(newtime, 0)
        newcube.add_aux_coord(newfcp, 0)
        newcube.attributes['STASH'] = dstStash
        newcube.attributes['Processed_accumulated'] = 'True'

        iris.save(newcube, ofile, zlib=True)
    try:
        os.remove(temp_accum_cubefile)
    except:
        pass

    if returncube:
        print('Cube saved to:', ofile)
        return newcube
    else:
        return ofile


def selectAnalysisDataFromMass(start_dt, end_dt, stash, lbproc=0, lblev=False, odir=None, returncube=False, overwrite=False):
    '''
    *** This function only works inside the Met Office (UKMO) ***
    Select from MASS the operational 'late' analysis
    :param start_dt: Start date/time (as datetime object)
    :param end_dt: End date/time (as datetime object)
    :param stash: single stash code (int format)
    :param lbproc: time processing code (0 by default)
    :param lblev: Either True, False or a list of levels. Best to get all relevant levels, then subset, so set to True
    :param odir: Better to not set this, and let the function put the global data in the default dir on scratch
    :param returncube: True or False
    :param overwrite: True or False
    :return: Either a list of cubes or a merged cube depending on the value of returncube
    '''
    import location_config as config

    # Change only these values ...
    analysis_incr = 6 # hours

    settings = config.load_location_settings()
    # Directory to save global analysis data to
    if not odir:
        odir = settings['scratchdir'] + 'ModelData/um_analysis/'

    # Make sure the directory exists
    if not os.path.isdir(odir):
        os.makedirs(odir)

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

    datetime_list = make_timeseries(start_dt, end_dt, analysis_incr)
    # Loop through all analysis times
    for this_dt in datetime_list:

        it = this_dt.strftime('%Y%m%dT%H%MZ')

        nowstamp = dt.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        ofile = pathlib.PurePath(odir, it + '_analysis_' + str(stash) + '_' + str(lbproc) + '.nc').as_posix()
        tmpfile = os.path.dirname(ofile) + '/tmp' + nowstamp + '.pp'
        # Global operational model
        # gl-mn : Main model run
        # gl-up : Updated model analysis (using more observations)
        collection = 'moose:/opfc/atm/global/prods/' + this_dt.strftime('%Y') + '.pp'
        # * because some files are called calc.pp
        massfilename = 'prods_op_gl-up_' + this_dt.strftime('%Y%m%d_%H') + '_000*'

        if (not os.path.isfile(ofile)) or overwrite:

            # Create a query template
            queryfn = os.path.dirname(ofile) + '/query' + nowstamp
            queryfile = open(queryfn, 'w')
            queryfile.write('begin\n')
            queryfile.write('  stash=' + str(stash) + '\n')
            queryfile.write('  lbproc=' + str(lbproc) + '\n')
            if lbproc == 128:
                queryfile.write('  period={3 hour}\n')
                queryfile.write('  interval_type=2\n')
            if lblev:
                queryfile.write('  lblev=(' + ','.join([str(lev) for lev in lblev]) + ')\n')
            queryfile.write('  pp_file=\'' + massfilename + '\'\n')
            queryfile.write('end\n')
            queryfile.close()

            try:
                # Now do the moo select
                print('Extracting Operational Analysis from MASS:', massfilename, '; stash:', str(stash), '; to:', ofile)
                getresult = subprocess.check_output(['moo', 'select', '-q', '-f', '-C', queryfn, collection, tmpfile])
                if os.path.isfile(queryfn):
                    os.remove(queryfn)
                cube = iris.load_cube(tmpfile)
                iris.save(cube, ofile, zlib=True)
                ofilelist.append(ofile)
                os.remove(tmpfile)

            except:
                if os.path.isfile(queryfn):
                    os.remove(queryfn)
                print('moo select failed for ' + it + ' ; stash: ' + str(stash))
                this_dt += dt.timedelta(hours=analysis_incr)
                continue

        elif os.path.isfile(ofile):
            ofilelist.append(ofile)
            print(ofile)
            print(it + ': File already exists on disk')
        else:
            print(it + ': Something else went wrong ... probably should check what')

        if returncube and os.path.isfile(ofile):
            # Try to load the data and return a cube or cubelist
            # Load the data
            try:
                cube = iris.load_cube(ofile)
            except:
                continue

            # bndpos = 0.5
            # if not cube.coord('time').has_bounds():
            #     cube.coord('time').guess_bounds(bound_position=bndpos)
            # if not cube.coord('forecast_period').has_bounds():
            #     cube.coord('forecast_period').guess_bounds(bound_position=bndpos)

            # If Convert units to mm per hour
            try:
                if cube.units == cf_units.Unit('kg m-2 s-1'):
                    cube.convert_units('kg m-2 h-1')
            except:
                print('Can\'t change units')

            try:
                ocubes.append(cube)
            except:
                print('No model data for ' + it)

    if returncube:
        # Possibly throw in a merge or concatenate_cube here ...
        ocube = ocubes.merge_cube()
        return ocube
    else:
        return ofilelist


def make_query_file(nowstamp, stash, proc, massfilename, odir, lblev=None):
    # Create a query template
    queryfn = odir + '/query' + nowstamp
    queryfile = open(queryfn, 'w')
    queryfile.write('begin\n')
    queryfile.write('  stash='+str(stash)+'\n')
    queryfile.write('  lbproc=' + str(proc) + '\n')
    if lblev:
        queryfile.write('  lblev=(' + ','.join([str(lev) for lev in lblev]) + ')\n')
    queryfile.write('  pp_file=\''+massfilename+'\'\n')
    queryfile.write('end\n')
    queryfile.close()

    return queryfn


def run_MASS_select(nowstamp, queryfn, collection, ofile, ofilelist):

    # Now do the moo select
    print('Extracting from MASS to: ', ofile)

    # If now is <15 hours since the initialisation time, there's a chance it is not on MASS yet, so don't write a '*.notarchived' file
    # NB: The SEA suite takes between 6 and 12 hours to run on research queues
    init_time = dt.datetime.strptime(os.path.basename(ofile).split('_')[0], '%Y%m%dT%H%MZ')
    now_time = dt.datetime.strptime(nowstamp, '%Y%m%d%H%M%S%f')

    tmpfile = os.path.dirname(ofile) + '/tmp' + nowstamp + '.pp'

    # Write an empty file so that other instances of this script know that something is happening, and can jump to the next file
    if not os.path.isfile(ofile):
        pathlib.Path(ofile).touch()

    attempts = 0
    while (os.stat(ofile).st_size == 0) and (attempts < 3):

        try:
            getresult = subprocess.run(['moo', 'select', '-q', '-f', '-C', queryfn, collection, tmpfile], stdout=subprocess.DEVNULL)
        except:
            attempts += 1
            continue

        if getresult.returncode == 0:
            try:
                cube = iris.load_cube(tmpfile)
                iris.save(cube, ofile) #, zlib=True)
            except:
                attempts += 1
                continue
        elif getresult.returncode == 2:
            print('Attempt: ' + str(attempts) + '. No file atoms are matched by query text file')
        else:
            print('Subprocess returned an exit code that I\'ve not considered yet')

        attempts += 1

    if os.stat(ofile).st_size > 0:
        ofilelist.append(ofile)
    else:
        os.remove(ofile)

    if os.path.isfile(queryfn):
        os.remove(queryfn)

    if os.path.isfile(tmpfile):
        os.remove(tmpfile)

    return ofilelist


def selectModelDataFromMASS(init_times, stash, odir='', domain='SEAsia', plotdomain=None, lbproc=None, lblev=False, choice='newest', modelid_searchtxt=None, returncube=False, overwrite=False):

    '''
    *** This function only works inside the Met Office (UKMO) ***
    For a list of initialisation times, get the relevant model data from MASS
    :param init_times: list of datetimes. Initialisation times as output by the getInitTimes function
    :param stash: integer. Code for the variable that we want to extract (only takes one at a time)
    :param odir: string. Better to leave this set to None so the default is used. Output directory for the data minus jobid.
    :param domain: string. Either 'SEAsia', 'Africa', or 'Global'. Used only to retrieve the correct jobid (not for spatial subsetting)
    :param plotdomain: list of floats. Contains [xmin, ymin, xmax, ymax]. Used only to determine the available models within a jobid
    :param lbproc: integer. Normally either 0 (instantaneous) or 128 (time averaged)
    :param lblev: Either True, False or a list of levels. Best to get all relevant levels, then subset, so set to True
    :param choice: string. Choose from ['newest', 'most_common', 'first']. Indicates how to select the jobid when more
            than one exists in the list. Default is 'newest'
    :param modelid_searchtxt: string (or list). Allows you to subset the list of available model_ids. Most likely options:
            'ga6', 'ga7', 'km4p4', 'indkm1p5', 'malkm1p5', 'phikm1p5', 'global-prods' (global operational),
            'africa-prods' (africa operational)
    :param returncube: boolean. Returns an iris.cube.CubeList object of the extracted data
    :param overwrite: boolean. Allows extracting the data again in case the file disk is currupted
    :return: Either a list of the files extracted or a iris.cube.CubeList
    '''
    import location_config as config
    settings = config.load_location_settings()
    # Directory to save model data to
    if not odir:
        odir = settings['scratchdir'] + 'ModelData/'

    # Check that the request is not split across different jobids
    if isinstance(init_times, list):
        jobid = most_common([getJobID_byDateTime(it_dt, domain, choice) for it_dt in init_times])
    else:
        jobid = getJobID_byDateTime(init_times, domain, choice)
        init_times = [init_times]

    odir = pathlib.PurePath(odir,jobid).as_posix()

    # print('Getting model data for ',bbox,'; jobid: ',jobid,' .... ')
    # print('   ... Saving to: ',odir)

    if not pathlib.Path(odir).is_dir():
        pathlib.Path(odir).mkdir(parents=True)

    # Set lblevs if not already done so
    if lblev:
        lblev = [10, 15, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 600, 700, 750, 850, 900, 925, 950, 1000]

    if returncube:
        ocubes = iris.cube.CubeList([])

    ofilelist = []

    for it_dt in init_times:

        # Get the model name(s) for this jobid
        if plotdomain:
            # TODO: May need to combine these two functions in the end
            modellist= getModels_bybox(plotdomain)['model_list']
            thesemodels = getModelID_byJobID(jobid, searchtxt=modellist)
            if modelid_searchtxt:
                thesemodels = [mod for mod in thesemodels if modelid_searchtxt in mod]
        else:
            thesemodels = getModelID_byJobID(jobid, searchtxt=modelid_searchtxt)

        for thismodel in thesemodels:

            it = it_dt.strftime('%Y%m%dT%H%MZ')
            print(it, thismodel, sep=': ')

            # Make sure we get the correct stash code for precip depending on the model
            if str(stash) in ['4201', '4203', '5216', '5226']:
                stash = getPrecipStash(thismodel, lbproc=lbproc, type='short')

            # Get the correct collection name and MASS filename
            if thismodel == 'global-prods':
                # Global operational or parallel suite model
                opfc_domain = jobid.split('-')[1]  # jobid can be either africa-opfc or africa-psuite42
                if 'psuite' in opfc_domain:
                    collection = 'moose:/opfc/atm/global/prods/' + opfc_domain + '.pp'
                else:
                    collection = 'moose:/opfc/atm/global/prods/' + it_dt.strftime('%Y') + '.pp'
                massfilename = 'prods_op_gl-mn*' + it_dt.strftime('%Y%m%d_%H') + '_*'

            elif thismodel == 'africa-prods':
                # Tropical Africa Operational model
                opfc_domain = jobid.split('-')[1]  # jobid can be either africa-opfc or africa-psuite42
                if 'psuite' in opfc_domain:
                    collection = 'moose:/opfc/atm/africa/prods/' + opfc_domain + '.pp'
                else:
                    collection = 'moose:/opfc/atm/africa/prods/'+it_dt.strftime('%Y')+'.pp'
                massfilename = 'prods_op_qa*' + it_dt.strftime('%Y%m%d_%H') + '_*'

            else:
                # All other research models
                collection = 'moose:/devfc/' + jobid + '/field.pp'
                massfilename = it + '_' + thismodel + '_pver*'

            if lbproc is None:
                # Get a list of all the possible lbprocs and extract them (usually it is 0 or 128)
                lbproc = get_lbproc_by_stash(stash, jobid)

            if isinstance(lbproc, int):
                lbproc = [lbproc]

            for proc in lbproc:
                nowstamp = dt.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
                ofile = pathlib.PurePath(odir, it + '_' + thismodel + '_' + str(stash) + '_' + str(proc) + '.nc').as_posix()

                if os.path.isfile(ofile):
                    try:
                        # If it is a file, make sure we can open it ...
                        test = iris.load_cube(ofile)
                        test = None
                        print(it + ': File already exists on disk')
                        ofilelist.append(ofile)
                    except:
                        queryfn = make_query_file(nowstamp, stash, proc, massfilename, os.path.dirname(ofile), lblev=None)
                        ofilelist = run_MASS_select(nowstamp, queryfn, collection, ofile, ofilelist)
                        if os.path.isfile(queryfn):
                            os.remove(queryfn)
                elif (not os.path.isfile(ofile)) or overwrite:
                    queryfn = make_query_file(nowstamp, stash, proc, massfilename, os.path.dirname(ofile), lblev=None)
                    ofilelist = run_MASS_select(nowstamp, queryfn, collection, ofile, ofilelist)
                    if os.path.isfile(queryfn):
                        os.remove(queryfn)
                else:
                    print(it + ': Something else went wrong ... probably should check what')

                if returncube and os.path.isfile(ofile):
                    # Try to load the data and return a cube or cubelist
                    # Load the data
                    if str(stash) == '4201' or str(stash) == '21104':
                        ofile = accumulated2sequential(ofile)

                    try:
                        cube = iris.load_cube(ofile)
                    except:
                        continue

                    try:
                        if lbproc == 0:
                            bndpos = 0.5
                        else:
                            bndpos = 1

                        if not cube.coord('time').has_bounds():
                            cube.coord('time').guess_bounds(bound_position=bndpos)
                        if not cube.coord('forecast_period').has_bounds():
                            cube.coord('forecast_period').guess_bounds(bound_position=bndpos)
                    except:
                        continue

                    # If Convert units to mm per hour
                    try:
                        if cube.units == cf_units.Unit('kg m-2 s-1'):
                            cube.convert_units('kg m-2 h-1')
                    except:
                        print('Can\'t change units')

                    try:
                        ocubes.append(cube)
                    except:
                        print('No model data for ' + it)

    if str(stash) == '4201' or str(stash) == '21104':
        new_ofilelist = []
        for ofile in ofilelist:
            new_ofile = accumulated2sequential(ofile)
            new_ofilelist.append(new_ofile)
        ofilelist = new_ofilelist

    if returncube:
        # Possibly throw in a merge or concatenate_cube here ...
        return(ocubes)
    else:
        return(ofilelist)


def getPrecipStash(model_id, lbproc=None, type='long'):

    if isinstance(lbproc, list):
        lbproc = lbproc[0]

    if "africa-prods" in model_id and lbproc == 128:
        outstash = 'm01s04i201' # this is the only stash in op_tafr that has lbproc=128
    elif not ("ga" in model_id or "global" in model_id):
        outstash = 'm01s04i203' # Large scale rain rate
    else:
        outstash = 'm01s05i216'  # Total Precip rate

    if type == 'short':
        return(int(outstash[5] + outstash[-3:]))
    elif type == 'long':
        return(outstash)
    else:
        return(outstash)


def calc_thermo_indices(temp, rh, index='lifted-index', returncube=True, ofile=None):
    '''
    Calculate some thermodynamic indices using metpy
    :param temp: iris.cube.Cube (or filename) of air temperature on pressure levels
    :param rh: iris.cube.Cube (or filename) of relative humidity on pressure levels
    :param index: string. Choose from ['lifted-index', 'showalter-index', 'k-index', 'total-totals']
    :param returncube: boolean. If False, returns a file name
    :param ofile: string. If specified, the function will save the iris cube to the specified file
    :return: filename or iris.cube.Cube
    '''

    import cf_units
    import xarray as xr
    import metpy.calc as mpcalc
    from metpy.units import units

    # Load the file into a cube if not already
    if not isinstance(temp, iris.cube.Cube):
        try:
            temp = iris.load_cube(temp)
        except:
            return None

    if not isinstance(rh, iris.cube.Cube):
        try:
            rh = iris.load_cube(rh)
        except:
            return None

    # Add latitude and longitude coordinate bounds for both cubes
    cnames = ['latitude', 'longitude']
    for cn in cnames:
        if not temp.coord(cn).has_bounds():
            temp.coord(cn).guess_bounds()

    for cn in cnames:
        if not rh.coord(cn).has_bounds():
            rh.coord(cn).guess_bounds()

    # Get all the time points that exist in both temp and RH fields, and make a time coordinate with them
    tpts = rh.coord('time').points[np.isin(rh.coord('time').points, temp.coord('time').points)]
    myu = rh.coord('time').units
    newtcoord = iris.coords.DimCoord(tpts, standard_name='time', units=rh.coord('time').units, var_name='time')

    # Check the order of the pressure levels (metpy needs the values to go from high to low)
    neworder_t = np.argsort(temp.coord('pressure').points)[::-1]
    neworder_h = np.argsort(rh.coord('pressure').points)[::-1]
    temp.coord('pressure').points = temp.coord('pressure').points[neworder_t]
    rh.coord('pressure').points = rh.coord('pressure').points[neworder_h]
    presspos = [i for i, coord in enumerate(temp.coords()) if coord.name() == 'pressure'][0]
    if presspos == 1:
        temp.data = temp.data[:, neworder_t, :, :]
        rh.data = rh.data[:, neworder_h, :, :]
    elif presspos == 0:
        temp.data = temp.data[neworder_t, :, :]
        rh.data = rh.data[neworder_h, :, :]
    else:
        print('Don\'t know what position the pressure coordinate is in the cube')
        pdb.set_trace()

    # Create an empty data array (NB: there's a small chance that the coordinate positions may differ between files)
    latpos = [i for i, coord in enumerate(temp.coords()) if coord.name() == 'latitude'][0]
    lonpos = [i for i, coord in enumerate(temp.coords()) if coord.name() == 'longitude'][0]
    try:
        timpos = [i for i, coord in enumerate(temp.coords()) if coord.name() == 'time'][0]
        emptydata = ma.empty((len(tpts), temp.shape[latpos], temp.shape[lonpos]))
    except:
        timpos = None
        emptydata = ma.empty((1, temp.shape[latpos], temp.shape[lonpos]))

    # Create an output cube to put the calculated data in
    cube = iris.cube.Cube(data=emptydata, long_name=index.replace('-', '_'), var_name=index.replace('-', '_'), units=cf_units.Unit('1'), attributes=temp.attributes, dim_coords_and_dims=[(newtcoord, 0), (temp.coord('latitude'), 1), (temp.coord('longitude'), 2)])
    cube.add_aux_coord(temp.coord('forecast_period'), data_dims=0)
    cube.add_aux_coord(temp.coord('forecast_reference_time'))

    # Loop through time coordinate
    for ti, tpt in enumerate(myu.num2date(tpts)):

        # Take a slice of the time from both cubes
        tslice = temp.extract(iris.Constraint(time=lambda t: t == tpt))
        hslice = rh.extract(iris.Constraint(time=lambda t: t == tpt))

        # Adjust RH to avoid divide by zero errors
        hslice.data[np.where(hslice.data <= 1)] = 0.01

        # Set the reference level and get the pressure coordinates
        if index == 'showalter-index':
            reference_level = 850.
        else:
            # used for the lifted index
            reference_level = 925.

        pvals = tslice.coord('pressure').points * units.hPa
        i = pvals.tolist().index(reference_level * units.hPa)

        temperature = xr.DataArray(tslice.units.convert(tslice.data.data, cf_units.Unit('degC')) * units.degC,
                                   dims=('pressure', 'latitude', 'longitude'),
                                   coords={'pressure': tslice.coord('pressure').points * units.hPa,
                                           'latitude': tslice.coord('latitude').points * units.degrees_north,
                                           'longitude': tslice.coord('longitude').points * units.degrees_east})
        humidity = xr.DataArray(hslice.data.data * units.percent,
                                dims=('pressure', 'latitude', 'longitude'),
                                coords={'pressure': hslice.coord('pressure').points * units.hPa,
                                        'latitude': hslice.coord('latitude').points * units.degrees_north,
                                        'longitude': hslice.coord('longitude').points * units.degrees_east})
        td = mpcalc.dewpoint_from_relative_humidity(temperature, humidity)

        t850 = temperature.metpy.sel(pressure=850.).metpy.unit_array
        t500 = temperature.metpy.sel(pressure=500.).metpy.unit_array
        td850 = td.metpy.sel(pressure=850.).metpy.unit_array
        t700 = temperature.metpy.sel(pressure=700.).metpy.unit_array
        td700 = td.metpy.sel(pressure=700.).metpy.unit_array

        if index == 'k-index':
            k_index = (t850 - t500) + td850 - (t700 - td700)
            cube.data[ti, :, :] = k_index.m

        if index == 'total-totals':
            total_totals = td850 - (2 * t500) + t850
            cube.data[ti, :, :] = total_totals.m

        # Unfortunately metpy doesn't calculate lifted index, or showalter on a grid, only 1D
        if index in ['showalter-index', 'lifted-index', 'CAPE', 'CIN']:
            for y, lat in enumerate(humidity.latitude.metpy.unit_array):
                for x, lon in enumerate(humidity.longitude.metpy.unit_array):
                    print(str(ti) + '/' + str(len(tpts)) + '; ' + str(y) + '/' + str(
                        len(hslice.coord('latitude').points)) + '; ' + str(x) + '/' + str(
                        len(hslice.coord('latitude').points)))
                    tprof = temperature.metpy.sel(latitude=lat.m, longitude=lon.m).metpy.unit_array
                    tdprof = td.metpy.sel(latitude=lat.m, longitude=lon.m).metpy.unit_array
                    tcell = temperature.metpy.sel(pressure=reference_level, latitude=lat.m,
                                                  longitude=lon.m).metpy.unit_array
                    tdcell = td.metpy.sel(pressure=reference_level, latitude=lat.m,
                                          longitude=lon.m).metpy.unit_array  # Running time: 27.5842 seconds (all)
                    parcel_profile = mpcalc.parcel_profile(pvals, tcell,
                                                           tdcell)  # Running time: 15.8242 seconds (91 grid cells)
                    if index in ['showalter-index', 'lifted-index']:
                        cube.data[ti, y, x] = mpcalc.lifted_index(pvals, tprof, parcel_profile).m
                    if index == 'CAPE':
                        cube.data[ti, y, x] = mpcalc.cape_cin(pvals, tprof, tdprof, parcel_profile)[0].m
                    if index == 'CIN':
                        cube.data[ti, y, x] = mpcalc.cape_cin(pvals, tprof, tdprof, parcel_profile)[1].m

    if ofile:
        iris.save(cube, ofile)

    if returncube:
        return cube
    else:
        return ofile


def compute_rh(q, t, p, es_eqtn='default'):
    """
    Calculates RH from air specific humidity, temperature and pressure
    RH is the ratio of actual water mixing ratio to saturation mixing ratio
    See Annex 4.A. pg184 of https://library.wmo.int/doc_num.php?explnum_id=10179
    and
    https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html
    and
    https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
    and
    http://cires1.colorado.edu/~voemel/vp.html
    :param q: float. Specific humidity (kg/kg) ratio of water mass / total air mass
    :param t: float. Temperature (degrees C)
    :param p: float. Pressure (hPa)
    :param es_eqtn: string. Either 'cc1', 'cc2', 'default' or empty. Determines which method is used for calculating
    saturation vapour pressure
    :return: float. Relative humidity as a percentage.
    """
    if not isinstance(q, np.ndarray) :
        q = np.array(q)
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    if not isinstance(p, np.ndarray):
        p = np.array(p)

    # molar mass of water vapour (g/mol)
    Mw = 18.01528
    # molar mass of dry air (g/mol)
    Md = 28.9634
    # Mixing ratio of moist air over water
    r = 0.62198 # Mw / Md # Approx 0.622
    # molar gas constant (J/molK)
    R = 8.3144621
    # specific gas constant for water vapour
    Rw = 1000 * R / Mw
    # Reference temperature (K)
    T0 = 273.15
    # Latent heat of evaporation for water (J/Kg)
    L = 2.5*np.power(10,6)
    # Saturation Vapour Pressure at the reference temperature (hPa)
    esT0 = 6.112

    # First convert degrees to Kelvin (or check the input data was not Kelvin anyway)
    if np.nanmin(t) > 100:
        tk = t
        t = t - T0
    else:
        tk = T0 + t

    # Saturation Vapour Pressure (hPa)
    # Lots of different ways of doing this, but they all seem to have fairly similar results.
    # In the end, I decided to use the WMO definition, but Clausius-Clapeyron worked just as well.

    if es_eqtn == 'cc1':
        # Using Clausius-Clapeyron
        es = esT0 * np.exp(L/Rw * ((1 / T0) - (1/tk)))
    elif es_eqtn == 'cc2':
        # Using Clausius-Clapeyron from Lawrence (2005) equation (10)
        # See https://journals.ametsoc.org/doi/pdf/10.1175/BAMS-86-2-225
        C2 = 2.53*np.power(10,9) # hPa at tk = 273.15
        es = C2 * np.exp((-L/Rw)/tk)
    else:
        # Using pg 188 of https://library.wmo.int/doc_num.php?explnum_id=10179
        # NB: 1.0016 is an enhancement factor proposed by Buck (1981) that is dependent on P and T
        #       for the tables, see Buck's table 1, or the introduction to Table 4.10 here:
        #       https://library.wmo.int/doc_num.php?explnum_id=7997
        fp = 1.0016 + (3.15 * np.power(10.,-6)*p) - (0.074/p)
        ewt = 6.112 * np.exp((17.62 * t) / (t + 243.12))
        es = fp * ewt

    # Vapour Pressure (hPa)
    # See Annex 4.A. pg184 of https://library.wmo.int/doc_num.php?explnum_id=10179
    e = (q * p) / ((q * (1 - r)) + r)

    rh = 100 * e / es
    # pdb.set_trace()
    if not isinstance(rh, np.ndarray):
        rh = np.array(rh)

    rh[rh > 100.] = 100.0
    rh[rh <= 0] = 0.1

    return np.round(rh, 2)

def compute_td(rh, t):
    '''
    Computes td given RH and t bases on Eq. 8 in Lawrence (2004).
    Coefficients are as stated in text, based on Alduchov and Eskridge (1986)
    Lawrence, M.G., 2004: The Relationship between Relative Humidity and the Dewpoint
    Temperature in Moist Air - A Simple Conversion and Applications. DOI:10.1175/BAMS-86-2-225
    :param rh: list or numpy array. Relative humidity as a percentage
    :param t: list or numpy array. Degrees celcius
    :return: numpy array of dewpoint temperature in degrees celcius
    '''

    if isinstance(rh, list):
        rh = np.array(rh)

    if isinstance(t, list):
        t = np.array(t)

    if np.nanmin(t) > 100:
        t = t - 273.15

    A1 = 17.625
    B1 = 243.04  # deg C

    # Stop divide by zero errors
    rh[rh <= 0.] = 0.01

    num = B1 * (np.log(rh / 100.) + A1 * t / (B1 + t))
    den = A1 - np.log(rh / 100.) - A1 * t / (B1 + t)

    return np.round(num / den, 1)


def compute_wdir(u, v):
    '''
    Calculate wind direction based on model u and v vectors
    As a reminder, see http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
    and
    https://stackoverflow.com/questions/21484558/how-to-calculate-wind-direction-from-u-and-v-wind-components-in-r
    :param u: list or numpy array. U wind vector
    :param v: list or numpy array. V wind vector
    :return: array of the same shape of wind direction in degrees
    '''
    if isinstance(u, list):
        u = np.array(u)

    if isinstance(v, list):
        v = np.array(v)

    wdir = np.mod(180 + np.rad2deg(np.arctan2(u, v)), 360)
    wdir[(u == 0) & (v == 0)] = np.nan

    return wdir


def compute_wspd(u, v, units='m/s', returncube=False):
    '''
    Calculate the wind speed given model u and v vectors
    As a reminder see http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
    :param u: list or numpy array. U wind vector
    :param v: list or numpy array. V wind vector
    :param units: string. 'knots' or 'm/s'
    :return: array of the same shape of wind speed
    '''

    cube = None

    if isinstance(u, iris.cube.Cube):
        cube = u.copy()
        u = u.data.copy()
        returncube = True

    if isinstance(v, iris.cube.Cube):
        cube = v.copy()
        v = v.data.copy()
        returncube = True

    if isinstance(u, list):
        u = np.array(u)

    if isinstance(v, list):
        v = np.array(v)

    if units == 'knots':
        c = 1.943844
    else:
        c = 1

    wspd = np.sqrt(u**2 + v**2) * c

    if returncube and cube:
        wspd_cube = cube.copy(data=wspd)
        wspd_cube.rename('Wind Speed')
        return wspd_cube
    else:
        return wspd


def compute_allwind(u_cube, v_cube, spd_levels=None):

    Y = np.repeat(u_cube.coord('latitude').points[..., np.newaxis], u_cube.shape[1], axis=1)
    X = np.repeat(u_cube.coord('longitude').points[np.newaxis, ...], u_cube.shape[0], axis=0)
    U = u_cube.data
    V = v_cube.data
    speed = np.sqrt(U * U + V * V)
    wdir = compute_wdir(U, V)

    if not isinstance(spd_levels, np.ndarray):
        lw = speedToLW(speed)
    else:
        lw = speedToLW(speed, maxspeed=spd_levels.max())

    return Y, X, U, V, speed, wdir, lw


def speedToLW(speed, maxspeed=None, maxlw=4):
    '''
    Function to convert windspeed into a sensible linewidth
    '''
    if maxspeed:
        return maxlw * speed / maxspeed
    else:
        return (0.5 + speed) / 4.


def LWToSpeed(lw, maxspeed=None):
    ''' The inverse of speedToLW, to get the speed back from the linewidth '''
    if maxspeed:
        return (lw * maxspeed) - 0.5
    else:
        return (lw * 4.) - 0.5


def makeStreamLegend(lx, spd_levels, color='k', fmt='{:.1f}'):

    # Turn off axes lines and ticks, and set axis limits
    lx.axis('off')
    lx.set_xlim(0, 1)
    lx.set_ylim(0, 1)
    nlines = len(spd_levels)
    lwds = speedToLW(spd_levels, maxspeed=spd_levels.max())

    for i, y in enumerate(np.linspace(0.1, 0.9, nlines)):

        # This line width
        lw = lwds[i]

        # Plot a line in the legend, of the correct width
        lx.axhline(y=y, xmin=0.1, xmax=0.4, c=color, lw=lw)

        # Add a text label, after converting the lw back to a speed
        lx.text(0.55, y, fmt.format(spd_levels[i]), va='center')

    lx.set_title('Wind Speed\n(m/s)')


def makeStreamLegend_fromLineCollection(lx, strm, nlines=4, color='k', fmt='{:.1f}', maxspeed=None):
    '''
    Make a legend for a streamplot on a separate axes instance
    :param strm: the output from a streamplot call
    :param lx: legend axes
    :param nlines:
    :param color:
    :param fmt:
    :return: nothing, but plots a legend
    '''

    # Get the linewidths from the streamplot LineCollection
    lws = np.array(strm.lines.get_linewidths())

    # Turn off axes lines and ticks, and set axis limits
    lx.axis('off')
    lx.set_xlim(0, 1)
    lx.set_ylim(0, 1)

    # Loop over the desired number of lines in the legend
    for i, y in enumerate(np.linspace(0.1, 0.9, nlines)):
        # This linewidth
        lw = lws.min() + float(i) * lws.ptp() / float(nlines - 1)

        # Plot a line in the legend, of the correct length
        lx.axhline(y=y, xmin=0.1, xmax=0.4, c=color, lw=lw)

        # Add a text label, after converting the lw back to a speed
        lx.text(0.5, y, fmt.format(LWToSpeed(lw, maxspeed=maxspeed)), va='center')

    lx.set_title('Wind Speed\n(m/s)')


def unpack_data2plot(data2plot, fieldname):
    '''
    Takes all the cubes with the same variable name out of data2plot
    :param data2plot:
    :param fieldname:
    :return:
    '''
    if 'wind' in fieldname:
        newfname = None
        for k1 in data2plot.keys():
            for k2 in data2plot[k1].keys():
                fieldlist = list(data2plot[k1][k2].keys())
                try:
                    newfname = [fn for fn in fieldlist if fieldname in fn][0]
                except:
                    continue
        if not newfname:
            print('Not able to find a wind fieldname')
        else:
            fieldname = newfname

    cubelist = iris.cube.CubeList([])
    for k1 in data2plot.keys():
        if fieldname in list(data2plot[k1].keys()):
            cube = data2plot[k1][fieldname]
            cubelist.append(cube)
        else:
            for k2 in data2plot[k1].keys():
                if fieldname in list(data2plot[k1][k2].keys()):
                    cube = data2plot[k1][k2][fieldname]
                    if cube:
                        cubelist.append(cube)
                else:
                    # pdb.set_trace()
                    # print('Not able to find fieldname')
                    continue

    return cubelist


def get_contour_levels(data2plot, fieldname, extend='neither', level_num=200):
    '''
    Get the min and max of the field for plotting contour levels. Requires a nested dictionary with 2 levels of keys before the fieldname keys
    :param data2plot: big dictionary with 2 levels of keys
    :param fieldname: the name of the field that we need to do the stats on. Can also be just 'wind', in whuch case, the wind speed is calculated
    :param extend: string. Same as matplotlib, one of [ 'neither' | 'both' | 'min' | 'max' ] if 'neither', then min and max of the data will be used. If extend='min' (or 'max' or 'both') then we calculate the value of the 1st percentile instead. This removes outliers so the colour scale has more variation.
    :param level_num: a higher number makes the colorbar look more continuous
    :return: an evenly array of numbers to use as contour levels
    '''

    if 'wind' in fieldname or 'speed' in fieldname:
        cubelist_uwind = unpack_data2plot(data2plot, 'Uwind')
        cubelist_vwind = unpack_data2plot(data2plot, 'Vwind')
        cubelist = iris.cube.CubeList([])
        for u, v in zip(cubelist_uwind, cubelist_vwind):
            wspeed = compute_wspd(u, v, returncube=True)
            cubelist.append(wspeed)
    else:
        cubelist = unpack_data2plot(data2plot, fieldname)

    try:
        cubem = cubelist.merge_cube()

        if extend == 'neither':
            fmin = cubem.data.min()
            fmax = cubem.data.max()
        elif extend == 'min':
            fmin = cubem.collapsed([c.name() for c in cubem.dim_coords], iris.analysis.PERCENTILE, percent=[1]).data.data[0]
            fmax = cubem.data.max()
        elif extend == 'max':
            fmin = cubem.data.min()
            fmax = cubem.collapsed([c.name() for c in cubem.dim_coords], iris.analysis.PERCENTILE, percent=[99]).data.data[0]
        elif extend == 'both':
            fmin = cubem.collapsed([c.name() for c in cubem.dim_coords], iris.analysis.PERCENTILE, percent=[1]).data.data[0]
            fmax = cubem.collapsed([c.name() for c in cubem.dim_coords], iris.analysis.PERCENTILE, percent=[99]).data.data[0]
        else:
            fmin = cubem.data.min()
            fmax = cubem.data.max()
    except:
        # We can't merge into a cube if there are multiple models in data2plot
        values = ma.masked_array([])
        for cube in cubelist:
            values = ma.append(values, cube.data)

        if extend == 'neither':
            fmin = values.min()
            fmax = values.max()
        elif extend == 'min':
            fmin = np.percentile(values, 1)
            fmax = values.max()
        elif extend == 'max':
            fmin = values.min()
            fmax = np.percentile(values, 99)
        elif extend == 'both':
            fmin = np.percentile(values, 1)
            fmax = np.percentile(values, 99)
        else:
            fmin = values.min()
            fmax = values.max()

    contour_levels = np.linspace(fmin, fmax, level_num)

    # if np.all(np.equal(contour_levels, 0)):
    #     contour_levels = np.linspace(0, 1, level_num)

    return contour_levels


def gpmLatencyDecider(inlatency, end_date):

    # Decide which latency to run the program with
    now = dt.datetime.utcnow()
    auto_latency = {'NRTearly': now - dt.timedelta(hours=3),
                    'NRTlate': now - dt.timedelta(hours=18),
                    'production': now - relativedelta(months=4)
                    }

    if inlatency == 'all':
        out_latency = ['production', 'NRTlate', 'NRTearly']
    elif inlatency == 'auto':
        best_latency = 'NRT_early'
        for l in auto_latency.keys():
            if end_date <= auto_latency[l]:
                best_latency = l
        out_latency = best_latency
    else:
        out_latency = inlatency

    return out_latency


def add_hour_of_day(cube, coord, name='hour'):
    add_categorised_coord(cube, name, coord, lambda coord, x: coord.units.num2date(x).hour)

def am_or_pm(cube, coord, name='am_or_pm'):
    add_categorised_coord(cube, name, coord, lambda coord, x: 'am' if x < 12 else 'pm')

def accum_6hr(cube, coord, name='6hourly'):
    add_categorised_coord(cube, name, coord, lambda coord, x: 0 if x < 6 else 1 if x < 12 else 2 if x < 18 else 3)

def accum_3hr(cube, coord, name='3hourly'):
    add_categorised_coord(cube, name, coord, lambda coord, x: 0 if x < 3 else 1 if x < 6 else 2 if x < 9 else 3 if x < 12 else 4 if x < 15 else 5 if x < 18 else 6 if x < 21 else 7)

def add_time_coords(cube):
    '''
    Adds some standard time coordinates to a cube for aggregation
    :param cube: An iris cube
    :return: An iris cube with more time coords
    '''
    # Add day of year, hour of day, category of 12hr or 6hr
    iris.coord_categorisation.add_day_of_year(cube, 'time', name='day_of_year')
    add_hour_of_day(cube, cube.coord('time'))
    am_or_pm(cube, cube.coord('hour'))
    accum_6hr(cube, cube.coord('hour'))
    accum_3hr(cube, cube.coord('hour'))

    return cube


def plot_country_etc(ax):

    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    borderlines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')
    ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
    ax.coastlines(resolution='50m', color='black')
    gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}


def plot_cube(cube, title=None, stretch=None, ofile=None):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib as mpl
    import iris.plot as iplt
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    mpl.rcParams["figure.figsize"] = [12.8, 9.6]
    fig = plt.figure(dpi=200)

    if stretch == 'low':
        pcm = iplt.pcolormesh(cube, norm=colors.PowerNorm(gamma=0.2))
    else:
        pcm = iplt.pcolormesh(cube)
        
    if title:
        plt.title(title)
    else:
        plt.title(cube.name())
    plt.xlabel('longitude / degrees')
    plt.ylabel('latitude / degrees')
    var_plt_ax = plt.gca()

    borderlines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')
    var_plt_ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
    var_plt_ax.coastlines(resolution='50m', color='black')
    gl = var_plt_ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
    gl.xlabels_top = False
    # gl.ylabels_left = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    vleft, vbottom, vwidth, vheight = var_plt_ax.get_position().bounds
    plt.gcf().subplots_adjust(top=vbottom + vheight, bottom=vbottom + 0.04,
                              left=vleft, right=vleft + vwidth)
    cbar_axes = fig.add_axes([vleft + 0.1, vbottom - 0.02, vwidth - 0.2, 0.02])
    cbar = plt.colorbar(pcm, cax=cbar_axes, orientation='horizontal', extend='both') # norm=norm, boundaries=bounds,
    # cbar.set_label(these_units)
    cbar.ax.tick_params(length=0)

    if ofile:
        fig.savefig(ofile, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_compare(cube1, cube2, filename=None):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import iris.plot as iplt

    # Get the bbox (xmin, xmax, ymin, ymax)
    xmin1, ymin1, xmax1, ymax1 = getCubeBBox(cube1, outtype='list')
    xmin2, ymin2, xmax2, ymax2 = getCubeBBox(cube2, outtype='list')
    domain = [min([xmin1, xmin2]), max([xmax1, xmax2]), min([ymin1, ymin2]), max([ymax1, ymax2])]

    fig = plt.figure(figsize=(15, 5))
    plt.subplot(131)
    try:
        cube1title = cube1.attributes['title']
    except KeyError:
        cube1title = cube1.name()
    plt.title(cube1title)
    pcm1 = iplt.pcolormesh(cube1, norm=colors.PowerNorm(gamma=0.5), cmap='PuBu')
    fig.colorbar(pcm1, extend='max', orientation='horizontal')
    ax = plt.gca()
    plot_country_etc(ax)
    ax.set_extent(domain)

    plt.subplot(132)
    try:
        cube2title = cube2.attributes['title']
    except KeyError:
        cube2title = cube2.name()
    plt.title(cube2title)
    pcm2 = iplt.pcolormesh(cube2, norm=colors.PowerNorm(gamma=0.5), cmap='PuBu')
    fig.colorbar(pcm2, extend='max', orientation='horizontal')
    ax = plt.gca()
    plot_country_etc(ax)
    ax.set_extent(domain)

    # Difference plot
    plt.subplot(133)
    cube3 = cube1 - cube2
    values = ma.getdata(cube3.data)
    valmax = np.round(np.max(np.array(np.abs(np.percentile(values, 1)), np.abs(np.percentile(values, 99)))))
    valmin = valmax * -1
    cube3title = 'Difference (cube1 - cube2)'
    plt.title(cube3title)
    pcm3 = iplt.pcolormesh(cube3, norm=colors.Normalize(vmin=valmin, vmax=valmax), cmap='PiYG')
    fig.colorbar(pcm3, extend='both', orientation='horizontal')
    ax = plt.gca()
    plot_country_etc(ax)
    ax.set_extent(domain)

    # plt.tight_layout()

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def make_nice_filename(file):
    '''
    Interprets a standard filename, and replaces stash and lbproc codes with nice text
    :param file: filename formatted e.g. 20200519T0300Z_analysis_3225_0.nc
    :return: filename with text instead of numbers
            e.g. 20200519T0300Z_analysis_Uwind10m_inst.nc
            e.g. 20200519T0300Z_SEA5-km4p4-ra2t_Uwind10m_inst.nc
    '''

    # Replaces the stash code with a nice name if it is in the stash code file
    stashdf = get_default_stash_proc_codes()

    # Tries to read the stash and lbproc code from the filename
    filebn = os.path.basename(file)
    datestr = filebn.split('_')[0]
    # Sometimes, there might be a region name in the filename, sometimes not
    # If there, it will be at the end of the filename e.g. _region.nc
    # So, we need to check if the -1th value converts to an int
    try:
        # fileproc = int(filebn.split('_')[-1].replace('.nc', ''))
        fileprocloc = -1
    except ValueError:
        fileprocloc = -2

    filestash = filebn.split('_')[fileprocloc - 1]
    fileproc = filebn.split('_')[fileprocloc].replace('.nc', '')
    model_id = filebn.split(datestr+'_')[1].split('_'+filestash)[0]
    new_model_id = model_id.replace('_', '-')
    file = file.replace(model_id, new_model_id)

    try:
        record = stashdf[(stashdf['stash'] == int(filestash)) & (stashdf['lbproc'] == int(fileproc))]
        file_nice = file.replace(
                            filestash + '_' + fileproc,
                            record['name'].to_string(index=False).lstrip(' ') + '_' + lut_lbproc(int(fileproc))
                            )
    except:
        file_nice = file

    return file_nice


def make_outputplot_filename(region_name, location_name, validtime, modelid, plot_location, timeagg, plottype, plotname, fclt, outtype='filesystem'):
    '''
    Given inputs, creates a filename in the standard format so it can be put into a webpage. This is important because the html creation code uses the values in the filename to create the menu structure
    Filename format:
    # <Plot-type>/<Valid-time>_<ModelId>_<Location>_<Time-Aggregation>_<Plot-Name>_<Lead-time>.png
    validtime_modelid_location_timeagg_plottype_forecastleadtime.png
    :param region_name: string. General region, in which multiple locations might exist. E.g. East Africa, SE Asia, etc
    :param location_name: string. Location of a zoom within the region, such as a case study location. E.g. Lake Victoria, Peninsular Malaysia, etc
    :param validtime: datetime. Time stamp for the file
    :param modelid: string. Name of the model or observation
    :param plot_location: string Name of the sub-domain that this plot is for
    :param timeagg: string. Time averaging period. This is most likely to be 'instantaneous', although it could also be '3hr', '6hr' etc
    :param plottype: string. A unique identifier for type of plot. This is typical the directory name e.g. precipitation
    :param plotname: string. A unique identifier for name of the plot within the plot type. e.g. precipitation-circulation, precipitation-postage-stamps, walker-circulation, etc
    :param fclt: string. Forecast lead time. Typically, this will be T+24, T+36, etc. If it is an observation, use T+0, or if all fclts are used, use something like 'All-FCLTs'
    :param outtype: string. Specifies whether to output a filename or a url
    :return: filename string or url
    '''
    import location_config as config
    settings = config.load_location_settings()
    region_name = region_name.replace(' ', '-').replace('_', '-').replace('/', '-')
    location_name = location_name.replace(' ', '-').replace('_', '-').replace('/', '-')
    plot_location = plot_location.replace(' ', '-').replace('_', '-').replace('/', '-')
    timedir = validtime.strftime('%Y%m')
    timestamp = validtime.strftime('%Y%m%dT%H%MZ')
    # 20210114T0600Z_analysis_Tropics-5S-to-5N_Instantaneous_walker-circulation_T+0.png
    base = region_name + '/' + location_name + '/' + plottype + '/' + timedir + '/' + timestamp + '_' + modelid + '_' + plot_location + '_' + timeagg + '_' + plotname + '_' + fclt + '.png'
    if outtype == 'filesystem':
        ofile = settings['plot_dir'].rstrip('/') + '/' + base

        # Make sure that the dir exists
        odir = os.path.dirname(ofile)
        if not os.path.isdir(odir):
            os.makedirs(odir)

    elif outtype == 'url':
        ofile = settings['url_base'].rstrip('/') + '/' + base
    else:
        return 'Didn\'t recognise outtype: ' + outtype


    return ofile


def send_to_ftp(filelist, ftp_path, settings, removeold=False):
    '''
    *** This function only works inside the Met Office (UKMO) ***
    Sends a list of local files to the ftp site, replacing the stash and lbproc codes with a nice name if possible
    :param filelist: list of local files
    :param ftp_path: folder on the ftp site (e.g. /)
    :param settings: dictionary read from the .config file
    :param removeold: boolean. If True, if a file on the ftp is not in the filelist, then delete it
    :return: print statements to show success or failure
    '''

    ftp_details = ['doftp', '-host', settings['ukmo_ftp_url'], '-user', settings['ukmo_ftp_user'], '-pass', settings['ukmo_ftp_pass']]
    stashdf = get_default_stash_proc_codes()

    dircheck = ftp_details.copy()
    dircheck.extend(['-mkdir', ftp_path])
    subprocess.run(dircheck)

    ftpfilecheck = ftp_details.copy()
    ftpfilecheck.extend(['-ls', ftp_path])
    attempts = 0
    while attempts < 5:
        try:
            result = subprocess.check_output(ftpfilecheck)
            break
        except:
            attempts += 1

    ftpfilelist = str(result).lstrip('\'b').rstrip('\\n\\n\'').split('\\n')

    if removeold:

        infiles = [os.path.basename(make_nice_filename(fn)) for fn in filelist]

        # Gets just the dates in the list of files to upload
        filedates = [fn.split('_')[0] for fn in infiles]

        # Gets just the model_stash_lbproc string from the list of files to upload
        modstashproc = list(set([fn[1+fn.find('_', 1):].rstrip('.nc') for fn in infiles]))

        # From the list of files on the ftp, this subsets according to the ones that match modstashproc
        ftpfiles_modstpr = [x for x in ftpfilelist if x[1+x.find('_', 1):].rstrip('.nc') in modstashproc]

        # From the modstashproc files that are on the ftp (ftpfiles_modstpr), which ones don't have dates in filedates?
        ftpfiles_todelete = [x for x in ftpfiles_modstpr if not os.path.basename(x).split('_')[0] in filedates]

        for ftpfile in ftpfiles_todelete:
            print('Removing ' + os.path.basename(ftpfile))
            ftpfiledel = ftp_details.copy()
            ftpfiledel.extend(['-delete', ftpfile])
            subprocess.check_output(ftpfiledel)

    for file in filelist:

        # TODO Check that the desired directory exists on the ftp, and if not, create it

        # This is probably not needed, but may not change things actually
        file_nice = make_nice_filename(file)

        if not os.path.basename(file_nice) in str(result):
            print('Sending ', file_nice)
            these_ftp_details = ftp_details.copy()
            try:
                these_ftp_details.extend(['-cwd', ftp_path, '-put', file + '=' + os.path.basename(file_nice)])
                subprocess.check_output(these_ftp_details)
            except:
                print(these_ftp_details, sep=' ')
                # pdb.set_trace()
                print('There was a problem sending', file_nice, 'to the FTP site')
        else:
            print(file_nice, 'already exists on FTP site')


def sort_pressure(cube):
    '''
    Operational global model output has non-monotonic sorting of the pressure field, which leads to problems later.
    This fixes those problems by sorting the pressure coordinate and data correctly
    :param cube: iris.cube.Cube
    :return: iris.cube.Cube
    '''

    # Take a copy of the input cube
    oldcube = cube.copy()

    # Initialise some variables
    new_data = None
    new_tpts = None
    new_fpts = None

    ind = np.lexsort((oldcube.coord('pressure').points, oldcube.coord('forecast_period').points, oldcube.coord('time').points))
    new_data = oldcube.data[ind, :, :].copy()

    # Apply the index to the pressure points so they match
    new_ppts = oldcube.coord('pressure').points[ind]
    # Only get pressure levels > 10hPa
    unique, counts = np.unique(new_ppts, return_counts=True)
    out_plevs = unique[counts == counts.max()]
    ppi = np.isin(new_ppts, out_plevs)
    new_tpts = oldcube.coord('time').points[ind][ppi]
    new_fpts = oldcube.coord('forecast_period').points[ind][ppi]
    new_ppts = oldcube.coord('pressure').points[ind][ppi]
    new_data = new_data[ppi, :, :]

    # Take a copy of the old cube coordinates
    oldp = oldcube.coord('pressure')
    oldt = oldcube.coord('time')
    oldf = oldcube.coord('forecast_period')
    oldlat = oldcube.coord('latitude')
    oldlon = oldcube.coord('longitude')

    # Create a new, correct, pressure coordinate
    pcoord = iris.coords.DimCoord(points=np.unique(new_ppts), units=oldp.units, long_name=oldp.long_name, var_name=oldp.var_name)
    tcoord = iris.coords.DimCoord(points=np.unique(new_tpts), units=oldt.units, long_name=oldt.long_name, var_name=oldt.var_name)
    fcoord = iris.coords.DimCoord(points=np.unique(new_fpts), units=oldf.units, long_name=oldf.long_name, var_name=oldf.var_name)

    # New data array
    new_data_4dims = ma.empty(shape=(len(np.unique(new_fpts)), len(np.unique(new_ppts)), len(oldlat.points), len(oldlon.points)))

    for i, fp in enumerate(np.unique(new_fpts)):
        dim_i, = np.where(new_fpts == fp)
        new_data_4dims[i, :, :, :] = new_data[np.newaxis, dim_i, :, :]

    # Use all of the above to create a new cube
    cube = iris.cube.Cube(data=new_data_4dims, standard_name=oldcube.standard_name, var_name=oldcube.var_name,
                          units=oldcube.units, attributes=oldcube.attributes,
                          dim_coords_and_dims=[(tcoord, 0), (pcoord, 1), (oldcube.coord('latitude'), 2),
                                               (oldcube.coord('longitude'), 3)])
    cube.add_aux_coord(fcoord, data_dims=0)
    cube.add_aux_coord(oldcube.coord('forecast_reference_time'))

    return cube


def get_time_segments(start, end, ta, max_plot_freq=12, start_hour_zero=True):
    '''
    Generates a list of tuples of start and end datetimes. The frequency (i.e. gap between tuples) is calculated either from ta (time aggregation period), or max_plot_freq, whichever is the smallest. For example, if ta=24 and max_plot_freq=12, then each tuple will be 24 hours from start to end, but that will be repeated every 12 hours.
    :param start: datetime for the start of the case study period
    :param end: datetime for the end of the case study period
    :param ta: integer. Time aggregation period
    :param max_plot_freq: integer. Maximum time difference between plots
    :return: list of tuples containing the start and end of each period
    '''
    step = min(ta, max_plot_freq)
    outlist = []

    # Make sure the start is anchored to a multiple of ta
    ## First, get a list of possible values
    if start_hour_zero:
        tmpstart = start.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        tmpstart = start
    xx = tmpstart
    poss_values = []
    while xx < tmpstart + dt.timedelta(days=1):
        poss_values.append(xx)
        xx = xx + dt.timedelta(hours=ta)
    ## Then, find the closest to the start
    stepstart = tmpstart
    while (stepstart + dt.timedelta(hours=ta)) <= start:
        stepstart += dt.timedelta(hours=ta)

    # Now, get the timeseries of tuples
    stepend = stepstart + dt.timedelta(hours=ta)
    while stepend <= end:
        outlist.append((stepstart, stepend))
        stepstart = stepstart + dt.timedelta(hours=step)
        stepend = stepstart + dt.timedelta(hours=ta)

    return outlist


def add_text_box_to_plot(text, figure=None, axes=None, fontsize=10):
    '''
    Adds an inset text box in the bottom right corner of a plot
    :param text: Text to put in the box
    :param figure: matplotlib figure object
    :param axes: matplotlib axes object within the figure
    :param fontsize: fontsize of the text
    :return: adds the text box to the plot
    '''

    from matplotlib.offsetbox import AnchoredText
    anchor = AnchoredText(text, prop=dict(size=fontsize), frameon=True, loc=4)
    anchor.patch.set_boxstyle("round, pad=0, rounding_size=0.2")
    axes = axes if axes else figure.gca()
    axes.add_artist(anchor)