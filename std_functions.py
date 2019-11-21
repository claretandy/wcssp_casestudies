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
        domain = None

    return domain

def getModels_bybox(plotdomain):

    xmin, ymin, xmax, ymax = plotdomain
    p1 = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymin)])

    domain_dict = {
        'SEAsia' : {
            'ga6'      : Polygon([(90, -18), (90, 30), (154, 30), (154, -18)]),
            'ga7': Polygon([(90, -18), (90, 30), (154, 30), (154, -18)]),
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
        }
    }
    model_list = []
    domain_list = []
    for k, v in domain_dict.items():
        for k1, v1 in v.items():
            if p1.intersects(v1):
                model_list.append(k1)
                domain_list.append(k)

    # Accounts for the fact that the global model matches everything
    domain = Counter(domain_list).most_common()[0][0]

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
                         'u-ba482': [dt.datetime(2018, 9, 6, 12, 0), dt.datetime.now()] # started earier, but the first date that ra1tld appears
                         }
    elif domain == 'TAfrica':
        dtrng_byjobid = {'u-ao907': [dt.datetime(2017, 7, 30, 12, 0), dt.datetime(2019, 4, 1, 0)],
                         'opfc': [dt.datetime(2019, 4, 1, 0), dt.datetime.now()]
                         }
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
            pdb.set_trace()
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

    return(outmodellist)

    
def getModelID_byDatetime(thisdate, domain='SEAsia', searchtxt=False):
    # print(searchtxt)
    jobid = getJobID_byDateTime(thisdate, domain=domain, choice='newest')
    modellist = getModelID_byJobID(jobid, searchtxt=searchtxt)

    return {"jobid": jobid, "modellist":modellist}


def getModelFileName(jobid, model_id):
    
    model_id_dict = {
        "ga7"     : "SEA2_n1280_ga7",
        "ga7plus" : "SEA3_n1280_ga7plus", #"SEA2_n1280_ga7plus",
        "ga6_sea1" : "SEA1_n1280_ga6",
        "ga6_sea2" : "SEA2_n1280_ga6",
        "ga6_sea3"     : "SEA3_n1280_ga6", #"SEA2_n1280_ga6",
        "km4p4_protora1"  : "SEA2_km4p4_protora1t",
        "km4p4_ra1": "SEA3_km4p4_protora1t", #"SEA2_km4p4_protora1t",
        "km4p4_4p0": "SEA1_km4p4_sing4p0",
        "km1p5_phil_ra1"  : "SEA3_phi2km1p5_protora1t", #"PHI2_km1p5_protora1t",
        "km1p5_phil_protora1t" : "PHI2_km1p5_protora1t",
        "km1p5_phil_4p0"  : "PHI_km1p5_vn4p0",
        "km1p5_mal_ra1"   : "SEA3_mal2km1p5_protora1t", #"MAL2_km1p5_protora1t",
        "km1p5_mal_protora1t" : "MAL2_km1p5_protora1t",
        "km1p5_mal_4p0"   : "MAL_km1p5_vn4p0",
        "km1p5_indon_ra1" : "SEA3_indon2km1p5_protora1t", #"INDON2_km1p5_protora1t",
        "km1p5_indon_4p0" : "INDON_km1p5_singv4p0",
        "km1p5_indon_protora1t" : "INDON2_km1p5_protora1t",
        "km1p5_singv_ra1" : "SINGRA1_km1p5ra1_protora1t",
        "km1p5_singv_4p1" : "SINGRA1_km1p54p1_v4p1",
        "km1p5_singv_4p0" : "SINGV3_km1p5_singv4p0"
        }
    return(model_id_dict[model_id])



def get1p5modelid(locname):
    # This refers to the 'getModelFileName' function below
    modelids1p5 = {
                'kuala_lumpur': 'km1p5_mal', #'mal_ra1',
                'manila'      : 'km1p5_phil', #'phil_ra1',
                'jakarta'     : 'km1p5_indon' #'indon_ra1'
        }
    #return(model_id + '_' + modelids1p5[locname])
    return(modelids1p5[locname])


def make_time_func(t1m, t2m):
    def tfunc(cell):
        return t1m < cell.point <= t2m
    return tfunc


def periodConstraint(cube, t1, t2):
    # Constrains the cube according to min and max datetimes
    #print(t1, ' to ', t2)
    timeUnits = cube.coord('time').units
    t1n = timeUnits.date2num(t1)
    t2n = timeUnits.date2num(t2)
    tfunc = make_time_func(t1, t2)
    tconst = iris.Constraint(time=tfunc)
    #print(cube.coord('time'))
    ocube = cube.extract(tconst)
    # pdb.set_trace()

    return(ocube)


def loadModelData(start, end, stash, plotdomain, searchtxt=None, lbproc=0, aggregate=True, overwrite=False):
    '''
    Loads all available model runs and clips data:
        - spatially (within lat/on box specified by plotdomain) and
        - temporally (between start and end datetimes)
    start : needs to be a datetime object
    end   : needs to be a datetime object
    stash : short format, as a string
    plotdomain : [xmin, ymin, xmax, ymax]
    timeagg    : 1, 3, 6, 12, 24 hours
    mod   : model_id
    jobid : jobid e.g. u-ba482
    odir  : typcially /scratch/hadhy/seasia
    lbproc: assumes we want 0 (instantaneous data), but 128, 4096 and 8192 are also possible
    overwrite: do we want to re-extract the model data from MASS? NB: the cropped data is not saved

    NB: old args:
    def loadModelData(start, end, stash, plotdomain, timeagg, model_id, jobid, odir, lbproc, overwrite=False)
    '''

    # 1. Get model domain for the given plotting domain
    domain = getDomain_bybox(plotdomain)
    if not domain:
        return "Not loading data because no data in the plotting domain"

    jobid = getJobID_byDateTime(end, domain=domain)
    model_id = getModelID_byJobID(jobid, searchtxt=searchtxt)
    model_id = model_id[0] if isinstance(model_id, list) else model_id
    odir = '/scratch/hadhy/'+domain.lower()+'/CaseStudies/'

    # 2. Extract from MASS
    # pdb.set_trace()
    if str(stash) in ['4201', '4203', '5216', '5226']:
        stash = getPrecipStash(model_id, type='short')

    print(model_id, jobid, stash, lbproc)

    cubelist = selectModelDataFromMASS(getInitTimes(start, end, domain, model_id=model_id), stash, odir, domain=domain, plotdomain=plotdomain, lbproc=lbproc, choice='newest',searchtxt=searchtxt, returncube=True, overwrite=overwrite)

    # 3. Loop through data and load into a cube list
    outcubelist = iris.cube.CubeList([])
    for cube in cubelist:
        # 3A. Subset to plot domain
        #print(cube)
        try:
            cube_dclipped = cube.intersection(latitude=(plotdomain[1],plotdomain[3]), longitude=(plotdomain[0],plotdomain[2]))
            # 3B. Extract time period
            # pdb.set_trace()
            cube_tclipped = periodConstraint(cube_dclipped, start, end)

            if not cube_tclipped:
                continue

            # 3C. Convert units
            try:
                if cube_tclipped.units == cf_units.Unit('kg m-2 s-1'):
                    cube_tclipped.convert_units('kg m-2 h-1')
            except:
                print("Can\'t change units")


            # 3D. Aggregate
            if aggregate:
                if len(cube_tclipped.coord('time').points) > 1:
                    cube_tclipped = cube_tclipped.collapsed('time', iris.analysis.MEAN) # Assuming precip
                    # diff_hrs = (end - start).total_seconds() // 3600
                    # print(diff_hrs)
                    # cube_tclipped.data = cube_tclipped.data * diff_hrs
                    cube_tclipped.attributes['aggregated'] = 'True'
                    cube_tclipped.attributes['units_note'] = 'Values represent mean rate over the aggregation period'
            else:
                cube_tclipped.attributes['units_note'] = 'Values represent those given by cube.units'


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
    gdal.RasterizeLayer(ds, [1], veclyr, options=["ATTRIBUTE="+attribute+""])

    # 3. Convert the resulting gdal dataset back to a cube
    ocube = geotiff2cube(ds)

    return ocube


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
        fcl = 72 # 3 days
    elif 'km4p4' in model_id or 'ga' in model_id:
        fcl = 120 # 5 days
    elif 'prod' in model_id:
        fcl = 54 # 2 and a bit days (Operational TAfrica)
    else:
        print('Guessing the forecast length as 120 hours')
        fcl = 120

    return(fcl)


def get_fc_InitHours(jobid):

    initdict = {
        'analysis' : [0, 6, 12, 18],
        'opfc' : [6,18],
        'u-ao907'   : [0,12],
        'u-ai613'   : [0,12],
        'u-ao347'   : [0,12],
        'u-ao902'   : [0,12],
        'u-ba482'   : [0,12]
    }
    try:
        init_hrs = initdict[jobid]
    except:
        print('Guessing 0Z and 12Z start times')
        init_hrs = [0,12]

    return(init_hrs)


def getInitTimes(start_dt, end_dt, domain, model_id=None, fcl=None, init_hrs=None, searchtxt=None, init_incr=None):
    '''
    start = start of the period for comparison to obs
    end   = end of the period for comparison to obs
    domain= name of the domain
    model_id= Name of model within the jobid (e.g. km4p4)
    fcl   = forecast length in DAYS (5 for 4.4km model, and 1.5 days for 1.5km models)
    init_hrs = list of the hours each day that the model is initialised
    init_incr = legacy option, not used
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


def getInitTimes_old(strt_dt, end_dt, fcl=None, init_incr=12):
    '''
    Replaced on 4th June 2019
    start = start of the period for comparison to obs
    end   = end of the period for comparison to obs
    fcl   = forecast length in DAYS (5 for 4.4km model, and 1.5 days for 1.5km models)
    '''
    # Most recent init date has to be before the start date!
    # Oldest init date has to be before the end date minus 5 days!
    #fcl = 120 # hours
    # init_incr = 12 # hours

    if not fcl:
        # Guess the forecast length from the start date (assume it doesn't change)
        jobid = getJobID_byDateTime(strt_dt, choice='newest')
        fcl = get_fc_length(jobid)

    # 1. Find the most recent init time BEFORE the end_dt
    mostrecent_init = end_dt.replace(hour=myround(end_dt.hour, base=init_incr), minute=0, second=0, microsecond=0) - dt.timedelta(hours=init_incr)

    # 2. Find the datetime of oldest model run available ...
    oldest_init = strt_dt.replace(hour=myround(strt_dt.hour, base=init_incr), minute=0, second=0, microsecond=0) + dt.timedelta(hours=init_incr) - dt.timedelta(hours=int(fcl))
    #print('Most recent init: ', mostrecent_init)
    #print('Oldest init: ', oldest_init)

    # 3. Loop through oldest to most recent init dates to create a timeseries
    init_ts = []

    while oldest_init <= mostrecent_init:
        init_ts.append(oldest_init) # .strftime('%Y%m%dT%H%MZ')
        oldest_init += dt.timedelta(hours=init_incr)

    return(init_ts)


def most_common(lst):
    return max(set(lst), key=lst.count)


def myround(x, base=3):
    return int(base * np.floor(float(x)/base))


def myroundup(x, base=3):
    return int(base * np.ceil(float(x)/base))


def get_lbproc_by_stash(stash, jobid):
    # which lbproc codes go with which stash codes?
    #TODO Maybe better to write an 'if' statement to separate out global (section 5) from convection permitting
    lbproc128_avail = {'u-ai613'  : [23, 4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205],
                         'u-ao347': [23, 4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205],
                         'u-ao902': [23, 4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205],
                         'u-ao907': [23, 4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205],
                         'u-ba482': [4201, 4202, 4203, 4204, 5201, 5202, 5205, 5206, 5216, 5226, 9202, 9203, 9204, 9205, 21100, 21104],
                         'opfc'   : []
                         }

    lbproc4096_avail = {'u-ai613' : [],
                         'u-ao347': [],
                         'u-ao902': [],
                         'u-ao907': [],
                         'u-ba482': [20080],
                         'opfc'   : []
                         }

    lbproc8192_avail = {'u-ai613' : [],
                         'u-ao347': [],
                         'u-ao902': [],
                         'u-ao907': [],
                         'u-ba482': [3463, 20080],
                        'opfc': []
                         }

    # 4201, 4202, 5201, 5202, 5226, 23
    lbproc0_notavail = {'u-ai613' : [],
                         'u-ao347': [],
                         'u-ao902': [],
                         'u-ao907': [4201, 4202, 5201, 5202, 5226, 23],
                         'u-ba482': [],
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

    # Directory to save global analysis data to
    if not odir:
        odir = '/scratch/hadhy/ModelData/UM_Analysis/'

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

    # Loop through all analysis times
    this_dt = start_dt
    while this_dt <= end_dt:

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

        this_dt += dt.timedelta(hours=analysis_incr)

    if returncube:
        # Possibly throw in a merge or concatenate_cube here ...
        ocube = ocubes.merge_cube()
        return (ocube)
    else:
        return (ofilelist)


def selectModelDataFromMASS(init_times, stash, odir, domain='SEAsia', plotdomain=None, lbproc=None, lblev=None, choice='newest',
                            searchtxt=None, returncube=False, overwrite=False):

    '''
    For a list of initialisation times, get the relevant model data from MASS
    init_times = datetime list of initialisation times
    stash = 4 or 5 digit stash code (only one at a time)
    odir  = Output directory for the data (minus the jobid)
    choice = ['newest', 'most_common', 'first'] # indicates how to select the jobid when more than one exists in the list. Default is 'newest'
    searchtxt = False # text string to indicate what model data is required (e.g. km4p4, indkm1p5, etc)
    '''

    # Check that the request is not split across different jobids
    if isinstance(init_times, list):
        jobid = most_common([getJobID_byDateTime(it_dt, domain, choice) for it_dt in init_times])
    else:
        jobid = getJobID_byDateTime(init_times, domain, choice)
        init_times = [init_times]

    if jobid == 'opfc':
        # Operational model MASS path
        moddomain = searchtxt.split('_')[0]
        collection = 'moose:/opfc/atm/'+moddomain+'/prods/year.pp'
    else:
        collection = 'moose:/devfc/'+jobid+'/field.pp'

    odir = pathlib.PurePath(odir,jobid).as_posix()

    print('Getting model data for ',domain,'; jobid: ',jobid,' .... ')
    print('   ... Saving to: ',odir)

    if not pathlib.Path(odir).is_dir():
        pathlib.Path(odir).mkdir(parents=True)

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

            # TODO: Check that this is not needed. The function 'getPrecipStash' should probably be run in the controlling script, so that this script just takes stash code and model_id
            # Make sure we get the correct stash code for precip depending on the model
            # if str(stash) in ['4201', '4203', '5216', '5226']:
            #     stash = getPrecipStash(thismodel, type='short')

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
                tmpfile = os.path.dirname(ofile) + '/tmp'+nowstamp+'.pp'
                defo_notonmass = os.path.isfile(ofile.replace('.nc', '.notonmass'))

                # if ( not os.path.isfile(ofile) and not defo_notonmass ) or overwrite:
                if (not os.path.isfile(ofile)) or overwrite:

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

                    try:
                        # Now do the moo select
                        print('Extracting from MASS: ', it, '; stash: ', str(stash), '; lbproc: ', str(proc))
                        getresult = subprocess.check_output(['moo', 'select', '-q', '-f', '-C', queryfn, collection, tmpfile])
                        cube = iris.load_cube(tmpfile)
                        iris.save(cube, ofile, zlib=True)
                        ofilelist.append(ofile)
                        os.remove(queryfn)
                        os.remove(tmpfile)

                        if defo_notonmass:
                            os.remove(ofile.replace('.nc', '.notonmass'))
                    except:
                        # open(ofile.replace('.nc', '.notonmass'), 'a').close()
                        # pdb.set_trace()
                        os.remove(queryfn)
                        print('moo select failed for ' + it + ' ; stash: '+str(stash))
                elif os.path.isfile(ofile):
                    ofilelist.append(ofile)
                    print(ofile)
                    print(it + ': File already exists on disk')
                else:
                    print(it + ': Something else went wrong ... probably should check what' )

                if returncube and os.path.isfile(ofile):
                    # Try to load the data and return a cube or cubelist
                    # Load the data
                    try:
                        cube = iris.load_cube(ofile)
                    except:
                        continue

                    if lbproc == 0:
                        bndpos = 0.5
                    else:
                        bndpos = 1

                    if not cube.coord('time').has_bounds():
                        cube.coord('time').guess_bounds(bound_position=bndpos)
                    if not cube.coord('forecast_period').has_bounds():
                        cube.coord('forecast_period').guess_bounds(bound_position=bndpos)

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

        