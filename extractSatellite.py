import os
import std_functions as sf
import datetime as dt
import location_config as config
import iris
import cf_units
from osgeo import osr, ogr, gdal
import pandas as pd
import numpy as np
import numpy.ma as ma
from shapely.geometry import Polygon
import subprocess
import glob
import pdb


def latlon2projected(x, y, dst_proj4string):

    source = osr.SpatialReference()
    source.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

    target = osr.SpatialReference()
    target.ImportFromProj4(dst_proj4string)

    transform = osr.CoordinateTransformation(source, target)
    point = ogr.CreateGeometryFromWkt("POINT ("+str(x)+" "+str(y)+")")

    point.Transform(transform)

    return point.GetX(), point.GetY()


def makeGDALds(arrbt, nx, ny, gt, srs, driver_name, outfile):

    if driver_name == 'GTiff':
        # Write to geotiff
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create( outfile, nx, ny, 1, gdal.GDT_Float32 )
    elif driver_name == 'MEM':
        driver = gdal.GetDriverByName('MEM')
        dataset = driver.Create('', nx, ny, 1, gdal.GDT_Float32)
    else:
        return 'Please enter a valid driver name'

    dataset.SetGeoTransform(gt)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(arrbt)
    dataset.GetRasterBand(1).SetNoDataValue(-9999)
    dataset.FlushCache()

    return dataset

def projectToLatLong(ifile, outfile, img_bnd_coords, proj4string):

    ####################################
    # Inputs
    # For details see http://fcm7/projects/SPS/browser/SPS/trunk/data/products/MSG_Products.nl
    ulx_ll, uly_ll, lrx_ll, lry_ll = img_bnd_coords  # Bounding coordinates in latlon, although the the image is in mercator projection!!!
    # epsg_code = 3395 # Mercator projection code of the imagery
    ####################################

    # For testing
    # ifile = '/data/users/hadhy/HyVic/Obs/OLR_noborders/EIAM50_201901210230.png'

    # Open the image file
    ds = gdal.Open(ifile)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    # Mask out pixel values with no data
    arr = ma.masked_less(arr, 4)
    arr = ma.masked_greater(arr, 254)
    # Apply equation to retrieve brightness temperatures. See Chamberlain et al. 2013 https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/met.1403
    arrbt = (-0.44 * (arr - 4.))+308.
    # Fill masked values with -9999 (because the Geotiff format doesn't accept masked arrays)
    ma.set_fill_value(arrbt, -9999)
    arrbt = arrbt.filled()

    # Set some image info ...
    ## Gets the bounding coordinates in Mercator projection
    ulx, uly = latlon2projected(ulx_ll, uly_ll, proj4string)
    lrx, lry = latlon2projected(lrx_ll, lry_ll, proj4string)
    ## Calculate geotransform object
    nx, ny  = [ds.RasterXSize, ds.RasterYSize]

    ## NB: The coordinates start in the top left corner (usually), so
    ## xDist needs to be positive, and yDist negative
    xDist = (lrx - ulx) / nx if lrx > ulx else -1 * (lrx - ulx) / nx
    yDist = -1 * (uly - lry) / ny if uly > lry else (uly - lry) / ny
    rtnX, rtnY = [0,0]
    gt = [ulx, xDist, rtnX, uly, rtnY, yDist]
    ## Projection information
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj4string)
    # srs.ImportFromEPSG(epsg_code) # Documentation says it is mercator

    # Create dataset in mercator projection
    memfile = ''
    ds_merc = makeGDALds(arrbt, nx, ny, gt, srs, 'MEM', memfile)

    # Regrid to latlon
    gdal.Warp(outfile, ds_merc, dstSRS='EPSG:4326', dstNodata=-9999)

    # Convert geotiff to a cube and save
    timestamp = os.path.basename(outfile).split('.')[0].split('_')[1]
    cube = sf.geotiff2cube(outfile, timestamp)

    return cube


def getOLR(start, end, satellite, area, productid, sat_scratch):
    '''
    Extracts satellite data from MASS for the date range
    :param satdir: The code name for the satellite (e.g. MSG or HIM)
    :param loc: The code for the location (e.g. AM = African Rift Valley)
    :param start_dt: Start date format %Y%m%d
    :param end_dt: End date format %Y%m%d
    :param odir: Output directory root
    :return: a cube of OLR data for the whole period
    '''

    ofilelist = []

    print('   Getting OLR filenames for ' + start.strftime("%Y-%m-%d") + ' to ' + end.strftime("%Y-%m-%d"))
    delta = dt.timedelta(days=1)
    this_dt = start
    while this_dt < end:

        print('   ' + this_dt.strftime("%Y-%m-%d"))

        scratch_tarfile = sat_scratch + '/' + productid + '_' + this_dt.strftime('%Y%m%d') + '.tar'

        if not os.path.exists(scratch_tarfile):
            print('   ... Extracting from MASS ...')
            mootarfile = 'moose:/adhoc/projects/autosatarchive/' + satellite + '/' + area + '/'+this_dt.strftime('%Y%m%d')+'.tar'
            try:
                getresult = subprocess.run(['moo', 'get', '-q', '-f', mootarfile, scratch_tarfile])
            except:
                continue

        print('   ... Extracting from the tar file to scratch')
        getresult = subprocess.check_output(['tar', '-C', sat_scratch, '-xvf', scratch_tarfile])

        # Select files we want to keep
        print('   ... Tidying up files ...')
        files = glob.glob(sat_scratch + '/' + satellite + '/' + area + '/'+this_dt.strftime('%Y%m%d') + '/' + '*')
        ncfiles = [fn for fn in files if os.path.basename(fn).startswith(productid) and fn.endswith('.nc')]
        pngfiles = [fn for fn in files if os.path.basename(fn).startswith(productid) and fn.endswith('.png')]
        jpgfiles = [fn for fn in files if os.path.basename(fn).startswith(productid) and fn.endswith('.jpg')]

        if any(ncfiles):
            ofilelist.extend(ncfiles)
        elif any(pngfiles):
            ofilelist.extend(pngfiles)
        elif any(jpgfiles):
            ofilelist.extend(jpgfiles)
        else:
            ofilelist = ofilelist

        for fn in files:
            if not fn in ofilelist:
                os.remove(fn)

        this_dt += delta

    return sorted(ofilelist)


def getAutoSatDetails(bbox):


    domain_df = pd.DataFrame(columns=['Satellite', 'area', 'area_name', 'ProductID', 'Projection', 'nx', 'ny', 'lon1', 'lat1', 'lon2', 'lat2'])
    domain_df = domain_df.append(pd.DataFrame({'Satellite': ['MSG'], 'area': ['AF'], 'area_name': ['Africa'], 'ProductID': ['EIAF51'], 'projection': ['Mercator'], 'nx': [768], 'ny': [768], 'lon1': [-46.362], 'lat1': [42.0], 'lon2': [46.362], 'lat2': [-42.0]}))
    domain_df = domain_df.append(pd.DataFrame({'Satellite': ['MSG'], 'area': ['AM'], 'area_name': ['Great Rift Valley'], 'ProductID': ['EIAM50'], 'projection': ['Mercator'], 'nx': [960], 'ny': [1100], 'lon1': [14.0], 'lat1': [14.0], 'lon2': [46.0], 'lat2': [-18.0]}))
    domain_df = domain_df.append(pd.DataFrame({'Satellite': ['MSG'], 'area': ['AO'], 'area_name': ['West Africa'], 'ProductID': ['EIAO51'],
         'projection': ['Mercator'], 'nx': [1999], 'ny': [1460], 'lon1': [-38.0], 'lat1': [38.0], 'lon2': [24.0], 'lat2': [-10.0]}))
    domain_df = domain_df.append(pd.DataFrame({'Satellite': ['MSG'], 'area': ['BZ'], 'area_name': ['Brazil'], 'ProductID': ['EIBZ51'],
         'projection': ['Mercator'], 'nx': [600], 'ny': [525], 'lon1': [-70.0], 'lat1': [0.0], 'lon2': [-20.0], 'lat2': [-40.0]}))
    domain_df = domain_df.append(pd.DataFrame({'Satellite': ['MSG'], 'area': ['EM'], 'area_name': ['Eastern Mediterranean'], 'ProductID': ['EIEM51'],
         'projection': ['Mercator'], 'nx': [1440], 'ny': [810], 'lon1': [17.1], 'lat1': [42.5], 'lon2': [48.85], 'lat2': [28.0]}))
    domain_df = domain_df.append(pd.DataFrame({'Satellite': ['HIM8'], 'area': ['IN'], 'area_name': ['Maritime Continent'], 'ProductID': ['LIIN50'],
         'projection': ['Equirectangular'], 'nx': [2688], 'ny': [2016], 'lon1': [90], 'lat1': [30], 'lon2': [154], 'lat2': [-18]}))

    # Projection information
    rsphere = 6378137
    req = 6378169
    rpol = 6356583.8
    h = 42164000 - req

    # From autosat code:
    # http://fcm7/projects/SPS/browser/SPS/trunk/src/python/Sps_Utilities/SpsMod_Utils.py
    projstring_dict = {'Geostationary': "+proj=geos +lon_0={!r} +h="+str(h)+" +a="+str(req)+" +b="+str(rpol)+" +units=m",
                  'Mercator': "+proj=merc +lon_0={!r} +a="+str(rsphere)+" +b="+str(rsphere)+" +k=1.0 +units=m",
                  'Equirectangular': "+proj=eqc +lon_0={!r} +ellps=WGS84"
                  }

    xmin, ymin, xmax, ymax = bbox
    p1 = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

    satellite, area, area_name, productid, projection = [None, None, None, None, None]

    for row in domain_df.itertuples():
        xmin, ymin, xmax, ymax = [row.lon1, row.lat2, row.lon2, row.lat1]
        rowpol = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

        if p1.intersects(rowpol):
            satellite = row.Satellite
            area = row.area
            area_name = row.area_name
            productid = row.ProductID
            projection = row.projection
            lon0 = 0  # row.lon2 - row.lon1
            proj4string = projstring_dict[projection].format(lon0)
            # ulx_ll, uly_ll, lrx_ll, lry_ll
            img_bnd_coords = [xmin, ymax, xmax, ymin]

    return satellite, area, area_name, productid, proj4string, img_bnd_coords



def main(start=None, end=None, region_name=None, location_name=None, bbox=None, model_ids=None, ftp_upload=None):

    settings = config.load_location_settings()
    if not start:
        start = settings['start']
    if not end:
        end = settings['end']
    if not region_name:
        region_name = settings['region_name']
    if not location_name:
        location_name = settings['location_name']
    if not bbox:
        bbox = settings['bbox']
    if not model_ids:
        model_ids = settings['model_ids']
    if not ftp_upload:
        ftp_upload = settings['ftp_upload']

    overwrite = True

    satellite, area, area_name, productid, proj4string, img_bnd_coords = getAutoSatDetails(bbox)
    sat_scratch = settings['scratchdir'].rstrip('/') + '/ModelData/autosat'
    sat_datadir = settings['datadir'].rstrip('/') + '/satellite_olr/' + region_name + '/' + location_name

    # Loop through dates
    delta = dt.timedelta(days=1)
    this_dt = start

    while this_dt <= end:

        print(this_dt.strftime('%Y-%m-%d'))
        ocube_fn = sat_datadir + '/' + this_dt.strftime('%Y%m') + '/' + productid + '_' + this_dt.strftime('%Y%m%d') + '.nc'

        if not os.path.isdir(os.path.dirname(ocube_fn)):
            os.makedirs(os.path.dirname(ocube_fn))

        if (not os.path.isfile(ocube_fn)):  # and (len(ifiles) > 0):

            print('   Create a daily netcdf file for ', this_dt.strftime('%Y-%m-%d'))

            # Extract the data from the MASS archive, and return a sorted list of files ...
            ifiles = getOLR(this_dt, this_dt + delta, satellite, area, productid, sat_scratch)

            # Now loop through ifiles
            cubes = iris.cube.CubeList([])
            for file in ifiles:

                outtiff = os.path.splitext(file)[0] + '_ll.tif'

                if file.endswith('.nc'):
                    # For SE Asia, netcdf files are already archived in lat/lon
                    cubetmp = iris.load_cube(file)
                    u = cf_units.Unit('hours since 1970-01-01 00:00:00', calendar=cf_units.CALENDAR_STANDARD)
                    timecoord = iris.coords.DimCoord(cubetmp.coord('time').points[0], standard_name='time', units=u)
                    array = cubetmp.data[np.newaxis, ...]
                    cubell = iris.cube.Cube(array, dim_coords_and_dims=[(timecoord, 0), (cubetmp.coord('latitude'), 1), (cubetmp.coord('longitude'), 2)])

                elif not os.path.exists(outtiff) or overwrite:
                    # Projects and converts png or jpg to geotiff
                    cubell = projectToLatLong(file, outtiff, img_bnd_coords, proj4string)

                else:
                    # If the projection and conversion are already done, convert to a cube
                    timestamp = os.path.basename(outtiff).split('.')[0].split('_')[1]
                    cubell = sf.gdalds2cube(outtiff, timestamp)

                # Add the resulting cube to a list of cubes for this day
                cubes.append(cubell)

            # Concatenate everything together into a cube
            cube = cubes.concatenate_cube()

            print('   ... Saving netcdf file')
            # The following saves as int16 (rather than float), which cuts the filesize by 50%, and means it saves much more quickly
            # cube.data = (cube * 10).data.astype(np.int16)
            iris.save(cube, ocube_fn, zlib=True)

            # Remove all  files
            if os.path.isfile(ocube_fn):
                print('    ... Removing temporary files')
                for fn in ifiles:
                    os.remove(fn)  # png, jpg or nc files
                    try:
                        os.remove(os.path.splitext(os.path.basename(fn))[0] + '_ll.tif')
                    except:
                        continue

        this_dt += delta


if __name__ == '__main__':
    main()
