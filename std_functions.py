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
from shapely.geometry import Polygon
from collections import Counter
from dateutil.relativedelta import relativedelta
import location_config as config
import pdb

def getDomain_bybox(plotdomain):

    xmin, ymin, xmax, ymax = plotdomain
    p1 = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymin)])

    seasia  = Polygon([(90, -18), (90, 30), (154, 30), (154, -18)])
    tafrica = Polygon([(-19, -12), (-19, 22), (52, 22), (52, -12)])

    if p1.intersects(seasia):
        domain = 'SEAsia'
    elif p1.intersects(tafrica):
        domain = 'TAfrica'
    else:
        print('The domain does not match any available convective scale models')
        domain = 'Global'

    return domain


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


def getModels_bybox(plotdomain, reg=None):

    xmin, ymin, xmax, ymax = plotdomain
    p1 = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymin)])

    if not reg:
        reg = getDomain_bybox(plotdomain)

    domain_dict = {
        'SEAsia' : {
            'ga6'     : Polygon([(90, -18), (90, 30), (154, 30), (154, -18)]),
            'ga7'     : Polygon([(90, -18), (90, 30), (154, 30), (154, -18)]),
            'km4p4'   : Polygon([(90, -18), (90, 30), (154, 30), (154, -18)]),
            'phikm1p5': Polygon([(116, 3), (116, 21), (132, 21), (132, 3)]),
            'viekm1p5': Polygon([(96.26825, 4.23075), (96.26825, 25.155752), (111.11825, 25.155752), (111.11825, 4.23075)]),
            'malkm1p5': Polygon([(95.5, -2.5), (95.5, 10.5), (108.5, 10.5), (108.5, -2.5)]),
            'indkm1p5': Polygon([(100, -15), (100, 1), (120, 1), (120, -15)])
        },
        'TAfrica' : {
            'ga6'      : Polygon([(-19, -12), (-19, 22), (52, 22), (52, -12)]),
            'ga7'      : Polygon([(-19, -12), (-19, 22), (52, 22), (52, -12)]),
            'takm4p4'  : Polygon([(-19, -12), (-19, 22), (52, 22), (52, -12)]),
            'eakm4p4'  : Polygon([(21.49, -20.52), (21.49, 17.48), (52, 17.48), (52, -20.52)]),
            'africa_prods' : Polygon([(-19, -12), (-19, 22), (52, 22), (52, -12)]),
            'global_prods' : Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)])
        },
        'Global': {
            'opfc': Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)])
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
    # domain = Counter(domain_list).most_common()[0][0]

    if not domain or model_list == []:
        print('The domain does not match any available convective scale model domains')
        pdb.set_trace()

    return {"domain" : domain, "model_list": model_list}


def getJobID_byDateTime(thisdate, domain='SEAsia', choice='newest'):
    # Prefered method here ...
    # Provide a date, and return either the newest OR the oldest running model_id
    # NB: Update this when new model versions are run

    if domain == 'SEAsia':
        dtrng_byjobid = {'u-ai613': [dt.datetime(2016, 12, 15, 12, 0), dt.datetime(2017, 7, 17, 12, 0)],
                         'u-ao347': [dt.datetime(2017, 7, 13, 12, 0), dt.datetime(2017, 8, 11, 0, 0)],
                         'u-ao902': [dt.datetime(2017, 7, 28, 12, 0), dt.datetime(2019, 1, 15, 0, 0)],
                         'u-ba482': [dt.datetime(2018, 9, 6, 12, 0), dt.datetime(2019, 10, 30, 0, 0)], # started earier, but the first date that ra1tld appears
                         'u-bn272': [dt.datetime(2019, 10, 2, 0, 0), dt.datetime(2020, 8, 16, 0, 0)],
                         'u-bw324': [dt.datetime(2020, 8, 15, 0, 0), dt.datetime.now()]
                         }
    elif domain == 'TAfrica':
        dtrng_byjobid = {'u-ao907': [dt.datetime(2017, 7, 30, 12, 0), dt.datetime(2019, 4, 1, 0)],
                         'opfc': [dt.datetime(2019, 4, 1, 0), dt.datetime.now()]
                         }
    elif domain == 'Global':
        dtrng_byjobid = {'opfc': [dt.datetime(2015, 1, 1, 0), dt.datetime.now()]}
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

    return(outjobid)

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
                                'SEA4_phikm0p5_ra1tld', 'SEA4_viekm1p5_ra1tld'], # NB: not all these are available from August 2018
                 'u-bn272': ['SEA5_n1280_ga7', 'SEA5_km4p4_ra2t', 'SEA5_indkm1p5_ra2t', 'SEA5_malkm1p5_ra2t', 'SEA5_phikm1p5_ra2t', 'SEA5_viekm1p5_ra2t'],
                 'u-bw324': ['SEA5_n1280_ga7', 'SEA5_km4p4_ra2t', 'SEA5_indkm1p5_ra2t', 'SEA5_malkm1p5_ra2t',
                                'SEA5_phikm1p5_ra2t', 'SEA5_viekm1p5_ra2t'],
                 'opfc': ['africa_prods', 'global_prods']
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
        return(outmodellist)

    
def getModelID_byDatetime(thisdate, domain='SEAsia', searchtxt=False):
    # print(searchtxt)
    jobid = getJobID_byDateTime(thisdate, domain=domain, choice='newest')
    modellist = getModelID_byJobID(jobid, searchtxt=searchtxt)

    return {"jobid": jobid, "modellist":modellist}


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

def periodConstraint(cube, t1, t2):
    # Constrains the cube according to min and max datetimes
    def make_time_func(t1m, t2m):
        def tfunc(cell):
            return t1m <= cell.point <= t2m
        return tfunc

    tfunc = make_time_func(t1, t2)
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
        cube.coord('time').guess_bounds()

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
        - spatially (within lat/on box specified by plotdomain) and
        - temporally (between start and end datetimes)
    :param start: datetime
    :param end: datetime
    :param stash: string (short format e.g. 4203)
    :param plotdomain: list (format [xmin, ymin, xmax, ymax] each value is a float)
    :param searchtxt: string (optional). Name of the model (e.g. km4p4, km1p5, ga6, ga7, opfc, etc)
    :param lbproc: int (optional). Assumes we want 0 (instantaneous data), but 128, 4096 and 8192 are also possible
    :param aggregate: boolean. Aggregate over the start-end period or not?
    :param totals: boolean. Do the values represent the total accumulated over the aggregation period?
    :param overwrite: Overwrite files downloaded from MASS?
    :return: CubeList of all available model runs between the start and end
    """

    # 1. Get model domain for the given plotting domain
    domain = getDomain_bybox(plotdomain)
    if not domain:
        print("Not loading data because no data in the plotting domain")
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
        # 3A. Subset to plot domain
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
            print(model_id, 'not in domain')

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



def poly2cube(shpfile, attribute, cube):
    '''
    Returns a cube with values for the given attribute from the shapefile
    :param shpfile: An ESRI *.shp file
    :param attribute: An integer formatted column in the shapefile table
    :param cube: An iris cube on which the polygons will be mapped on to
    :return: A cube with values for the given attribute
    '''

    from osgeo import ogr
    from osgeo import gdal

    # 1. create a gdal dataset from the cube
    ds = cube2gdalds(cube)
    # print(ds.GetRasterBand(1).ReadAsArray().max())

    # 2. Rasterize the vectors
    vec = ogr.Open(shpfile)
    veclyr = vec.GetLayer()
    # fieldnames = [field.name for field in veclyr.schema]
    # print(fieldnames)
    # print(ds.GetGeoTransform())

    if attribute == '':
        gdal.RasterizeLayer(ds, [1], veclyr)
    else:
        gdal.RasterizeLayer(ds, [1], veclyr, options=["ATTRIBUTE=" + attribute + ""])

    # 3. Convert the resulting gdal dataset back to a cube
    ocube = geotiff2cube(ds)
    ds = None

    return ocube


def domainClip(cube, domain):
    '''
    Clips a cube according to a bounding box
    :param cube: An iris cube
    :param domain: list containing xmin, ymin, xmax, ymax or dictionary defining each
    :return: iris cube containing the clipped domain
    '''

    if isinstance(domain, dict):
        lonce = iris.coords.CoordExtent('longitude', domain['xmin'], domain['xmax'])
        latce = iris.coords.CoordExtent('latitude', domain['ymin'], domain['ymax'])
    else:
        xmin, ymin, xmax, ymax = domain
        lonce = iris.coords.CoordExtent('longitude', xmin, xmax)
        latce = iris.coords.CoordExtent('latitude', ymin, ymax)

    cube_cropped = cube.intersection(lonce, latce)

    return cube_cropped


def cube2gdalds(cube):
    '''
    :param cube: A 2D cube with bounds
    :return: A GDAL dataset
    '''
    from osgeo import gdal

    if not cube.coord('latitude').has_bounds():
        cube.coord('latitude').guess_bounds()
    if not cube.coord('longitude').has_bounds():
        cube.coord('longitude').guess_bounds()

    output = 'my.tif' # May not need this

    x_pixel_width = np.round(np.mean([x[1] - x[0] for x in cube.coord('longitude').bounds]), 6)
    y_pixel_width = np.round(np.mean([x[1] - x[0] for x in cube.coord('latitude').bounds]), 6)
    x_res = cube.coord('longitude').points.shape[0]
    y_res = cube.coord('latitude').points.shape[0]
    x_min = np.round(np.min([np.min(x) for x in cube.coord('longitude').bounds]), 5)
    y_max = np.round(np.max([np.max(y) for y in cube.coord('latitude').bounds]), 5)

    ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((x_min, x_pixel_width, 0, y_max, 0, y_pixel_width))
    band = ds.GetRasterBand(1)
    NoData_value = -999999
    band.SetNoDataValue(NoData_value)
    band.FlushCache()

    return ds

def geotiff2cube(geotiff, timestamp=None):

    '''
    :param geotiff: Input can be either a geotiff file or a gdal dataset object
    :param timestamp: Datetime (as a string formatted %Y%m%d%H%M)
    :return: iris cube
    '''

    from osgeo import gdal

    if isinstance(geotiff, str):
        ds = gdal.Open(geotiff)
    elif isinstance(geotiff, gdal.Dataset):
        ds = geotiff
    else:
        sys.exit('Didn\'t recognise the input to geotiff2cube')

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
    if nodatavalue:
        array = ma.masked_equal(array, nodatavalue)

    if timestamp:
        u = cf_units.Unit('hours since 1970-01-01 00:00:00', calendar=cf_units.CALENDAR_STANDARD)
        timecoord = iris.coords.DimCoord(u.date2num(dt.datetime.strptime(timestamp, '%Y%m%d%H%M')), standard_name='time', units=u)
        array = array[np.newaxis, ...]
        cube = iris.cube.Cube(array, dim_coords_and_dims=[(timecoord, 0), (latcoord, 1), (loncoord, 2)])

    else:
        # pdb.set_trace()
        cube = iris.cube.Cube(array, dim_coords_and_dims=[(latcoord, 0), (loncoord, 1)])

    return cube

def getOLR(satdir, loc, start_dt, end_dt, odir=None):
    '''
    Checks on /data/AutosatArchive/ImageArchive/Retrieve/andrew.hartley_001 first
    Then checks for satellite data from MASS for the date range
    :param satdir: The code name for the satellite (e.g. MSG or HIM)
    :param loc: The code for the location (e.g. AM = African Rift Valley)
    :param start_dt: Start date format %Y%m%d
    :param end_dt: End date format %Y%m%d
    :param odir: Output directory root
    :return: a cube of OLR data for the whole period
    '''
    #
    import shutil

    if not odir:
        odir = '/data/users/hadhy/HyVic/Obs/OLR_noborders/'
    diskdir = '/data/AutosatArchive/ImageArchive/Retrieve/andrew.hartley_001/'
    imgcode = 'EI'+loc+'50'

    delta = dt.timedelta(days=1)
    this_dt = start_dt
    while this_dt <= end_dt:
        print (this_dt.strftime("%Y-%m-%d"))
        diskfile = diskdir + imgcode + '_' + this_dt.strftime('%Y%m%d') + '.tar'
        if os.path.exists(diskfile):
            print('Extracting from the tar file')
            getresult = subprocess.check_output(['tar', '-C', odir, '-xvf', diskfile])
        else:
            print('Extracting from MASS ...')
            itarfile = 'moose:/adhoc/projects/autosatarchive/' + satdir + '/' + loc + '/'+this_dt.strftime('%Y%m%d')+'.tar'
            otarfile = odir + this_dt.strftime('%Y%m%d')+'.tar'
            try:
                getresult = subprocess.check_output(['moo', 'get', '-q', '-f', itarfile, odir])
                print('Extracting from archived file')
                result2 = subprocess.check_output(['tar', '-C', odir, '-xvf', otarfile])
            except:
                print('Unable to extract data for this day')

            # Move files into the big directory
            print('Tidying up files ...')
            for fn in glob.glob(odir + satdir + '/' + loc + '/' + '*.png'):
                 if os.path.basename(fn).startswith('EIAM50'):
                     os.rename(fn, odir + os.path.basename(fn))

            try:
                shutil.rmtree(odir + satdir + '/' + loc)
            except:
                print('File not found')

        this_dt += delta


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
    # print(cube.coord('latitude').bounds[0], cube.coord('latitude').bounds[-1])
    # print(ymin, ymax)

    if cube.coord('longitude').bounds[0][0] > cube.coord('longitude').bounds[0][1]:
        k, l = [0, 1]
    else:
        k, l = [1, 0]
    xmin = np.min([bnd[l] for bnd in cube.coord('longitude').bounds])
    xmax = np.max([bnd[k] for bnd in cube.coord('longitude').bounds])
    # print(cube.coord('longitude').bounds[0], cube.coord('longitude').bounds[-1])
    # print(xmin, xmax)

    if outtype == 'list':
        return [xmin, ymin, xmax, ymax]
    elif outtype == 'polygon':
        return Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymin)])
    else:
        return [xmin, ymin, xmax, ymax]

def get_fc_length(model_id):
    '''
    :param model_id: This is the ID of the model within the job, as a string
    :return: The length of model run in hours
    '''
    # print(model_id)

    if isinstance(model_id, list):
        model_id = model_id[0]

    if 'km1p5' in model_id:
        fcl = 48 # 2 days ... actually this changes depending on the jobid
    elif 'km4p4' in model_id or 'ga' in model_id:
        fcl = 120 # 5 days
    elif 'prod' in model_id:
        fcl = 168 # Operational Global
        # fcl = 54 # 2 and a bit days (Operational TAfrica)
    else:
        print('Guessing the forecast length as 120 hours')
        fcl = 120

    return(fcl)


def get_default_stash_proc_codes(list_type='long'):
    '''
    Returns a pandas dataframe of the
    :param list_type: 'short' or 'long' or 'profile'
    :return: pandas dataframe
    '''
    import pandas as pd

    if list_type in ['long', 'short', 'profile', 'share']:
        list_type = list_type + '_list'
    else:
        return 'List type does not exist'

    df = pd.read_csv('std_stashcodes.csv')
    outdf = df[df[list_type]]

    return outdf


def get_fc_InitHours(jobid):

    initdict = {
        'analysis' : [0, 6, 12, 18],
        'opfc' : [0,6,12,18],
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


def getInitTimes(start_dt, end_dt, domain, model_id=None, fcl=None, init_hrs=None, searchtxt=None, init_incr=None):
    '''
    Given a start and end date of a case study period, what are all the possible model runs that could be available?
    :param start_dt: datetime
    :param end_dt: datetime
    :param domain: string. Can be one of 'SEAsia', 'TAfrica' or 'Global'. This is used with the date to get the jobid
    :param model_id: string (optional). Name of the model within the jobid. See the function 'getModelID_byJobID', but generally
            can be one of 'ga6', 'ga7', 'km4p4', 'indkm1p5', 'malkm1p5', 'phikm1p5',
            or 'global_prods' (for the operational global)
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
        fcl = get_fc_length(model_id)

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
                       'opfc'   : []
                       }

    lbproc4096_avail = {'u-ai613': [],
                        'u-ao347': [],
                        'u-ao902': [],
                        'u-ao907': [],
                        'u-ba482': [20080],
                        'u-bn272': [20080],
                        'u-bw324': [20080],
                        'opfc'   : []
                        }

    lbproc8192_avail = {'u-ai613': [],
                        'u-ao347': [],
                        'u-ao902': [],
                        'u-ao907': [],
                        'u-ba482': [3463, 20080],
                        'u-bn272': [3463, 20080],
                        'u-bw324': [3463, 20080],
                        'opfc': []
                        }

    # 4201, 4202, 5201, 5202, 5226, 23
    lbproc0_notavail = {'u-ai613': [],
                        'u-ao347': [],
                        'u-ao902': [],
                        'u-ao907': [4201, 4202, 5201, 5202, 5226, 23],
                        'u-ba482': [],
                        'u-bn272': [],
                        'u-bw324': [],
                        'opfc'   : []
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
    #
    # # use a dictionary reader so we can access by field name
    # reader = csv.DictReader(open("volcano_data.txt", "rb"),
    #                         delimiter='\t',
    #                         quoting=csv.QUOTE_NONE)

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
    try:
        already_processed = cube.attributes['Processed_accumulated'] in ['True', 'true', 'TRUE']
    except KeyError:
        already_processed = False

    if already_processed:
        newcube = cube
        ofile = accum_cubefile
    else:
        srcStash = cube.attributes['STASH']
        dstStash = [sconv[1] for sconv in stash_convert if sconv[0] == srcStash]
        dstStash = srcStash if dstStash == [] else dstStash[0]

        latcoord = cube.coord('latitude')
        loncoord = cube.coord('longitude')

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
            # print(i, repr(x_slice))
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

        # pdb.set_trace()
        ofile = accum_cubefile.replace(str(srcStash[1])+str(srcStash[2]), str(dstStash[1])+str(dstStash[2]))
        iris.save(newcube, ofile, zlib=True)

    if returncube:
        print('Cube saved to:',ofile)
        return(newcube)
    else:
        return(ofile)


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

    # Change only these values ...
    analysis_incr = 6 # hours

    #########
    # Change nothing from here onwards

    settings = config.load_location_settings('UKMO')
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
        massfilename = 'prods_op_gl-up_' + this_dt.strftime('%Y%m%d_%H') + '_000.pp'

        if (not os.path.isfile(ofile)) or overwrite:

            # Create a query template
            queryfn = 'query' + nowstamp
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
                print('Extracting from MASS: ', it, '; stash: ', str(stash))
                getresult = subprocess.check_output(['moo', 'select', '-q', '-f', '-C', queryfn, collection, tmpfile])
                cube = iris.load_cube(tmpfile)
                iris.save(cube, ofile, zlib=True)
                ofilelist.append(ofile)
                os.remove(queryfn)
                os.remove(tmpfile)

            except:
                os.remove(queryfn)
                # os.remove(tmpfile)
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
        return (ocube)
    else:
        return (ofilelist)


def make_query_file(nowstamp, stash, proc, massfilename, lblev=None):
    # Create a query template
    queryfn = 'query' + nowstamp
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
    not_archived = ofile.replace('.nc', '.notarchived')

    if os.path.isfile(ofile):
        os.remove(ofile)

    attempts = 0
    while (not os.path.isfile(ofile)) and (not os.path.isfile(not_archived)) and (attempts < 3):

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

    if os.path.isfile(ofile):
        ofilelist.append(ofile)
    else:
        if init_time < (now_time - dt.timedelta(hours=15)):
            open(not_archived, 'a').close()

    os.remove(queryfn)
    if os.path.isfile(tmpfile):
        os.remove(tmpfile)

    return ofilelist


def selectModelDataFromMASS(init_times, stash, odir='', domain='SEAsia', plotdomain=None, lbproc=None, lblev=False, choice='newest', searchtxt=None, returncube=False, overwrite=False):

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
    :param searchtxt: string (or list). Allows you to subset the list of available model_ids. Most likely options:
            'ga6', 'ga7', 'km4p4', 'indkm1p5', 'malkm1p5', 'phikm1p5', 'global_prods' (global operational),
            'africa_prods' (africa operational)
    :param returncube: boolean. Returns an iris.cube.CubeList object of the extracted data
    :param overwrite: boolean. Allows extracting the data again in case the file disk is currupted
    :return: Either a list of the files extracted or a iris.cube.CubeList
    '''

    settings = config.load_location_settings('UKMO')
    # Directory to save model data to
    if not odir:
        odir = settings['scratchdir'] + 'ModelData/'

    # Check that the request is not split across different jobids
    if isinstance(init_times, list):
        jobid = most_common([getJobID_byDateTime(it_dt, domain, choice) for it_dt in init_times])
    else:
        jobid = getJobID_byDateTime(init_times, domain, choice)
        init_times = [init_times]

    if jobid == 'opfc':
        # Operational model MASS path
        moddomain = searchtxt.split('_')[0] # either africa_prods or global_prods
        collection = 'moose:/opfc/atm/'+moddomain+'/prods/year.pp'
    else:
        collection = 'moose:/devfc/'+jobid+'/field.pp'

    odir = pathlib.PurePath(odir,jobid).as_posix()

    # print('Getting model data for ',domain,'; jobid: ',jobid,' .... ')
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
            if searchtxt:
                thesemodels = [mod for mod in thesemodels if searchtxt in mod]
        else:
            thesemodels = getModelID_byJobID(jobid, searchtxt=searchtxt)

        for thismodel in thesemodels:
            # pdb.set_trace()
            it = it_dt.strftime('%Y%m%dT%H%MZ')
            print(it, thismodel, sep=': ')

            # Make sure we get the correct stash code for precip depending on the model
            if str(stash) in ['4201', '4203', '5216', '5226']:
                stash = getPrecipStash(thismodel, type='short')

            if jobid == 'opfc':
                # Replace collection path with correct year
                collection = collection.replace('year', it_dt.strftime('%Y'))

            if thismodel == 'global_prods':
                # Global operational model
                massfilename = 'prods_op_gl-mn*' + it_dt.strftime('%Y%m%d_%H') + '_*'
            elif thismodel == 'africa_prods':
                # Tropical Africa Operational model
                massfilename = 'prods_op_qa*' + it_dt.strftime('%Y%m%d_%H') + '_*'
            else:
                # All research models
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
                        queryfn = make_query_file(nowstamp, stash, proc, massfilename, lblev=None)
                        ofilelist = run_MASS_select(nowstamp, queryfn, collection, ofile, ofilelist)
                elif (not os.path.isfile(ofile)) or overwrite:
                    queryfn = make_query_file(nowstamp, stash, proc, massfilename, lblev=None)
                    ofilelist = run_MASS_select(nowstamp, queryfn, collection, ofile, ofilelist)

                else:
                    print(it + ': Something else went wrong ... probably should check what')

                if returncube and os.path.isfile(ofile):
                    # Try to load the data and return a cube or cubelist
                    # Load the data
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

    if returncube:
        # Possibly throw in a merge or concatenate_cube here ...
        return(ocubes)
    else:
        return(ofilelist)


def getPrecipStash(model_id, lbproc=None, type='long'):

    if "africa_prods" in model_id and lbproc == 128:
        # pdb.set_trace()
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


def compute_wspd(u, v, units='knots'):
    '''
    Calculate the wind speed given model u and v vectors
    As a reminder see http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
    :param u: list or numpy array. U wind vector
    :param v: list or numpy array. V wind vector
    :param units: string. 'knots' or 'm/s'
    :return: array of the same shape of wind speed
    '''
    if isinstance(u, list):
        u = np.array(u)

    if isinstance(v, list):
        v = np.array(v)

    if units == 'knots':
        c = 1.943844
    else:
        c = 1

    wspd = np.sqrt(u**2 + v**2) * c

    return wspd


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


def getGPMCube(start, end, latency, plotdomain, settings, aggregate=True):
    '''
    Creates a mean rainfall rate for the period defined by start and end, clips to a domain, and outputs a cubelist
    containing the data and quality flag
    plotdomain = xmin, ymin, xmax, ymax
    '''

    inpath = settings['gpm_path'] + 'netcdf/imerg/'+latency+'/'

    if start > end:
        raise ValueError('You provided a start_date that comes after the end_date.')

    # Gets all the filenames that are needed
    alldatafilelist = [glob.glob((start + dt.timedelta(days=x)).strftime(inpath + '%Y/gpm_imerg_' + latency + '_*_%Y%m%d*.nc')) for x in range(0, 1 + (end - start).days)]
    datafilelist = [item for sublist in alldatafilelist for item in sublist if not 'quality' in item]
    qualfilelist = [item for sublist in alldatafilelist for item in sublist if 'quality' in item]

    # Load the files as cubes
    datacubelist = iris.load(datafilelist)
    datacube = datacubelist.concatenate_cube()
    try:
        qualcubelist = iris.load(qualfilelist)
        qualcube = qualcubelist.concatenate_cube()
    except:
        qualcube = None

    # Clip to bounding box
    datacube = datacube.intersection(latitude=(plotdomain[1],plotdomain[3]), longitude=(plotdomain[0],plotdomain[2]))
    try:
        qualcube = qualcube.intersection(latitude=(plotdomain[1],plotdomain[3]), longitude=(plotdomain[0],plotdomain[2]))
    except:
        qualcube = None
    # Extract time period that we're interested in
    datacube = periodConstraint(datacube, start, end)
    if qualcube:
        qualcube = periodConstraint(qualcube, start, end)
    # Mask zero in preparation for aggregation
    if type(datacube.data) == np.ndarray:
        datacube.data = ma.masked_less(datacube.data, 0)
    else:
        datacube.data = ma.masked_less(datacube.data.data, 0)

    if qualcube:
        if type(qualcube.data) == np.ndarray:
            qualcube.data = ma.masked_less(qualcube.data, 0)
        else:
            qualcube.data = ma.masked_less(qualcube.data.data, 0)

    # Aggregate remaining time periods
    # NB: GPM data is recorded as a rate in mm/hour at 30 min intervals,
    #       therefore, when aggregating the precip totals need to be divided by 2
    if aggregate:
        datacube = datacube.collapsed('time', iris.analysis.SUM)
        datacube.data = datacube.data / 2.
        if qualcube:
            qualcube = qualcube.collapsed('time', iris.analysis.MEAN)
            qualcube.coord('latitude').guess_bounds()
            qualcube.coord('longitude').guess_bounds()

    if not datacube.coord('latitude').has_bounds():
        datacube.coord('latitude').guess_bounds()
        datacube.coord('longitude').guess_bounds()

    # Add metadata
    if qualcube:
        metalist = [datacube, qualcube]
    else:
        metalist = [datacube]

    for cube in metalist:
        cube.attributes['STASH'] = iris.fileformats.pp.STASH(1, 5, 216)
        cube.attributes['data_source'] = 'GPM'
        cube.attributes['product_name'] = 'imerg'
        cube.attributes['latency'] = latency
        cube.attributes['version'] = os.path.basename(glob.glob(datafilelist[0])[0]).split('_')[3]
        cube.attributes['aggregated'] = str(aggregate)
        cube.attributes['title'] = 'GPM '+latency

    if qualcube:
        qualcube.attributes['title'] = qualcube.attributes['title'] + ' Quality Flag'

    if qualcube:
        return(datacube, qualcube)
    else:
        return(datacube, None)


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
    cbar_axes = fig.add_axes([vleft, vbottom - 0.02, vwidth, 0.02])
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

    # Get the domain (xmin, xmax, ymin, ymax)
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

    plt.tight_layout()

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)

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
        fileproc = int(filebn.split('_')[-1].replace('.nc', ''))
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

def make_outputplot_filename(event_name, validtime, modelid, location, timeagg, plottype, plotname, fclt, outtype='filesystem'):
    '''
    Given inputs, creates a filename in the standard format so it can be put into a webpage. This is important because the html creation code uses the values in the filename to create the menu structure
    Filename format:
    # <Plot-type>/<Valid-time>_<ModelId>_<Location>_<Time-Aggregation>_<Plot-Name>_<Lead-time>.png
    validtime_modelid_location_timeagg_plottype_forecastleadtime.png
    :param event_name: string. Usual format: region/event (e.g. PeninsulaMalaysia/20200520_Johor )
    :param validtime: string. Formatted datetime
    :param modelid: string. Name of the model or observation
    :param location: string. Place name. If it contains spaces, fill with a '-' (NOT an underscore as this will mess up the webpage creation)
    :param timeagg: string. Time averaging period. This is most likely to be 'instantaneous', although it could also be '3hr', '6hr' etc
    :param plottype: string. A unique identifier for type of plot. This is typical the directory name e.g. precipitation
    :param plotname: string. A unique identifier for name of the plot within the plot type. e.g. precipitation-circulation, precipitation-postage-stamps, walker-circulation, etc
    :param fclt: string. Forecast lead time. Typically, this will be T+24, T+36, etc. If it is an observation, use T+0, or if all fclts are used, use something like 'All-FCLTs'
    :param outtype: string. Specifies whether to output a filename or a url
    :return: filename string or url
    '''

    settings = config.load_location_settings()
    location = location.replace(' ', '-')
    location = location.replace('/', '|')
    base = event_name + '/' + plottype + '/' + validtime + '_' + modelid + '_' + location + '_' + timeagg + '_' + plotname + '_' + fclt + '.png'
    if outtype == 'filesystem':
        ofile = settings['plot_dir'] + base

        # Make sure that the dir exists
        odir = os.path.dirname(ofile)
        if not os.path.isdir(odir):
            os.makedirs(odir)

    elif outtype == 'url':
        ofile = settings['url_base'] + base
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
