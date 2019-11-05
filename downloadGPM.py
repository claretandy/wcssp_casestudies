import iris
import cf_units
import glob
import numpy as np
import numpy.ma as ma
import os, sys
import errno
from datetime import timedelta, date, datetime
import datetime
import h5py
import pdb
#from readGPM_v2 import *
#iris.FUTURE.netcdf_no_unlimited = True

'''
Downloads GPM data into an iris cube.

'''

def getTimeCoord(file):

    timeunit = cf_units.Unit('hours since 1970-01-01', calendar = cf_units.CALENDAR_GREGORIAN)

    dtstrings = os.path.basename(file).split('.')[4]

    start_dt = datetime.datetime.strptime(dtstrings.split('-')[0] + '-' + dtstrings.split('-')[1], '%Y%m%d-S%H%M%S')
    end_dt = datetime.datetime.strptime(dtstrings.split('-')[0] + '-' + dtstrings.split('-')[2], '%Y%m%d-E%H%M%S')
    id_pt = start_dt + ((end_dt - start_dt)/2)

    timecoord = iris.coords.DimCoord([timeunit.date2num(id_pt)], bounds=[(timeunit.date2num(start_dt), timeunit.date2num(end_dt))], standard_name='time', units=timeunit)

    return(timecoord)

def mergeGPM(ifiles, ofile, year, month, day, var, version, latency):

    try:
        f00 = h5py.File(ifiles[0], 'r')
    except:
        print('No files found')
        return

    # Setup output cube
    geog_cs = iris.coord_systems.GeogCS(semi_major_axis=6378137, inverse_flattening=298.257223563) 

    latcoord = iris.coords.DimCoord(list(f00['Grid/lat']), standard_name='latitude', units='degrees', coord_system=geog_cs)
    loncoord = iris.coords.DimCoord(list(f00['Grid/lon']), standard_name='longitude', units='degrees', coord_system=geog_cs)
    precip = np.empty((len(ifiles), 1800, 3600))

    f00.close()
    gpm_cubelist = iris.cube.CubeList([])
    for file in sorted(ifiles):
        print(file)
        #pdb.set_trace()
        f00 = h5py.File(file, 'r')
        filedata = list(f00['Grid/' + var])
        precip = np.dstack(filedata)
        timecoord = getTimeCoord(file)
        try:
            cube = iris.cube.Cube(precip,
                            standard_name='precipitation_flux',
                            units='mm h-1',
                            attributes=None,
                            dim_coords_and_dims=[(timecoord, 0),(latcoord, 1),(loncoord, 2)])
        except ValueError:
            cube = iris.cube.Cube(precip,
                            standard_name='precipitation_flux',
                            units='mm h-1',
                            attributes=None,
                            dim_coords_and_dims=[(timecoord, 2),(latcoord, 1),(loncoord, 0)])
        except:
            continue

        mycoords = [c.name() for c in cube.coords()]
        if mycoords != ['time', 'latitude', 'longitude']:
            cube.transpose([mycoords.index('time'), mycoords.index('latitude'), mycoords.index('longitude')])

        gpm_cubelist.append(cube)
        f00.close()

    new_cube = gpm_cubelist.concatenate_cube()
    iris.save(new_cube, ofile, zlib=True)
    

def getGenericField(ifiles, var, year, month, day):
    
    field_names = {'IRkalmanFilterWeight': 'ir_weight', 
               'precipitationQualityIndex': 'quality_index'}
    
    try:
        f00 = h5py.File(ifiles[0], 'r')
    except:
        print('No files found')
        return

    # Setup output cube
    geog_cs = iris.coord_systems.GeogCS(semi_major_axis=6378137, inverse_flattening=298.257223563) 

    latcoord = iris.coords.DimCoord(list(f00['Grid/lat']), standard_name='latitude', units='degrees', coord_system=geog_cs)
    loncoord = iris.coords.DimCoord(list(f00['Grid/lon']), standard_name='longitude', units='degrees', coord_system=geog_cs)
    precip = np.empty((len(ifiles), 1800, 3600))

    f00.close()
    gpm_cubelist = iris.cube.CubeList([])
    for file in sorted(ifiles):
        #print(file)
        #pdb.set_trace()
        f00 = h5py.File(file, 'r')
        filedata = list(f00['Grid/' + var])
        precip = np.dstack(filedata)
        timecoord = getTimeCoord(file)
        try:
            cube = iris.cube.Cube(precip,
                                long_name=field_names[var],
                                attributes=None,
                                dim_coords_and_dims=[(timecoord, 0),(latcoord, 1),(loncoord, 2)])
        except ValueError:
            cube = iris.cube.Cube(precip,
                                long_name=field_names[var],
                                attributes=None,
                                dim_coords_and_dims=[(timecoord, 2), (latcoord, 1), (loncoord, 0)])
        except:
            continue

        mycoords = [c.name() for c in cube.coords()]
        if mycoords != ['time', 'latitude', 'longitude']:
            cube.transpose([mycoords.index('time'), mycoords.index('latitude'), mycoords.index('longitude')])

        gpm_cubelist.append(cube)
        f00.close()
        
    new_cube = gpm_cubelist.concatenate()
    return(new_cube)


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def getYMD(indate):
    year = indate.strftime("%Y")
    month = indate.strftime("%m")
    day = indate.strftime("%d")
    return(year, month, day)


def calcQuality(rawdatafiles, ofileq, year, month, day, curVer, latency):
    '''
    Extracts relevant information from the downloaded raw data files and 
    calculates a quality index that accounts for the amount of influence IR imagery has had on the precipitation estimate
    
    Results are scaled between 0-100 integers to reduce disk storage costs
    '''
    
    try:
        ir_weight, = getGenericField(rawdatafiles, 'IRkalmanFilterWeight', year, month, day)
        qualindex, = getGenericField(rawdatafiles, 'precipitationQualityIndex', year, month, day)
        
        mask = (ir_weight.data > 0.0)
        qualindex.data[mask] = qualindex.data[mask] * ((100-ir_weight.data[mask])/100.0)
        qualindex.data = ma.masked_less(qualindex.data, 0)
        qualindex.data = np.around(qualindex.data * 100).astype(np.uint8)
        
        qualindex.data.set_fill_value(0)
        
        iris.save(qualindex, ofileq, zlib=True)

    except:
        print('Error creating quality flag: ' + curVer)
    
    
def main(latency, days, end_date, outdir, agency):
    product = 'imerg' # This shouldn't change
    #change the accounts
    username_lookup = {'ukmo':'andrew.hartley@metoffice.gov.uk',
                       'pagasa':'username@pagasa.gov.ph',
                       }
    server = {'production' : ['arthurhou.pps.eosdis.nasa.gov', username_lookup[agency], '.HDF5'],
                'NRTlate' : ['jsimpson.pps.eosdis.nasa.gov', username_lookup[agency], '.RT-H5'],
                'NRTearly' : ['jsimpson.pps.eosdis.nasa.gov', username_lookup[agency], '.RT-H5']}
    var = 'precipitationCal'

    # Shouldn't need to change anything below here ..
    first_date = date(2000, 6, 1)
    #end_date = date.today()
    start_date = end_date - timedelta(days)
    if start_date < first_date:
        start_date = first_date

    # Loop through dates
    for single_date in daterange(start_date, end_date):

        print(latency + ' : ' + single_date.strftime("%Y-%m-%d"))
        
        year, month, day = getYMD(single_date)
        sfilepath = {'production' : '/gpmdata/'+year+'/'+month+'/'+day+'/imerg/3B-HHR.MS.MRG.3IMERG.',
                        'NRTlate' : '/NRTPUB/imerg/late/'+year+month+'/3B-HHR-L.MS.MRG.3IMERG.'+year+month+day,
                        'NRTearly': '/NRTPUB/imerg/early/'+year+month+'/3B-HHR-E.MS.MRG.3IMERG.'+year+month+day} # +'.*.RT-H5'
        # change '/project/earthobs/PRECIPITATION/GPM/rawdata/
        rawdata_dir = outdir+'/rawdata/'+product+'/'+latency+'/'+year+'/'+month+'/'+day
        netcdf_dir = outdir+'/netcdf/'+product+'/'+latency+'/'+year+'/'
        ofile_test = netcdf_dir + '/gpm_'+product+'_'+latency+'_*_'+year+month+day+'.nc'
        ofile_part_test = netcdf_dir + '/gpm_'+product+'_'+latency+'_*_'+year+month+day+'_part.nc'
        ofileq_test = netcdf_dir + '/gpm_'+product+'_'+latency+'_*_'+year+month+day+'_quality.nc'
        ofileq_part_test = netcdf_dir + '/gpm_'+product+'_'+latency+'_*_'+year+month+day+'_quality_part.nc'
        
        if not os.path.isdir(rawdata_dir):
            mkdir_p(rawdata_dir)
        if not os.path.isdir(netcdf_dir):
            mkdir_p(netcdf_dir)
        rawdatafiles = glob.glob(rawdata_dir + '/3B-HHR*' + server[latency][2])

        if ((len(glob.glob(ofile_test)) == 1) or (len(glob.glob(ofile_part_test)) == 2)) & (len(glob.glob(ofileq_test)) == 1) or (len(glob.glob(ofileq_part_test)) == 2):
            print('    Nothing to do')
        else:
            # Do everything
            if not len(rawdatafiles) == 48:
                thiswd = os.getcwd()
                os.chdir(rawdata_dir)
                print('    Downloading files from FTP ...')
                #pdb.set_trace()
                downloadstring = {
                        'ukmo':'export HTTP_PROXY=http://webproxy.metoffice.gov.uk:8080 ; /opt/ukmo/utils/bin/doftp -host '+server[latency][0] +' -user '+server[latency][1]+' -pass '+server[latency][1]+' -mget '+sfilepath[latency],
                        'pagasa':'ftp -host '+server[latency][0] +' -user '+server[latency][1]+' -pass '+server[latency][1]+' -mget '+sfilepath[latency]'
                        }
                
                os.system(downloadstring[agency])
                os.chdir(thiswd)

            rawdatafiles = glob.glob(rawdata_dir + '/3B-HHR*' + server[latency][2])
                
            versions = [os.path.basename(rdf).split('.')[6] for rdf in rawdatafiles]

            for curVer in set(versions):

                print('   ',curVer)

                ofile = netcdf_dir + 'gpm_'+product+'_'+latency+'_'+curVer+'_'+year+month+day+'.nc'
                ofile_part = netcdf_dir + 'gpm_'+product+'_'+latency+'_'+curVer+'_'+year+month+day+'_part.nc'

                ofileq = netcdf_dir + 'gpm_'+product+'_'+latency+'_'+curVer+'_'+year+month+day+'_quality.nc'
                ofileq_part = netcdf_dir + 'gpm_'+product+'_'+latency+'_'+curVer+'_'+year+month+day+'_quality_part.nc'

                if os.path.isfile(ofile) & os.path.isfile(ofileq) :

                    print('      Nothing to do for this version')

                else:

                    rawdatafiles = glob.glob(rawdata_dir + '/3B-HHR*' + curVer + server[latency][2])

                    # If we now have a complete list of netcdf files, make the daily netcdf
                    if len(rawdatafiles) == 48:
                        print('      Merging all 48 files ...')
                        mergeGPM(rawdatafiles, ofile, year, month, day, var, curVer, latency)
                        if os.path.isfile(ofile_part):
                            os.remove(ofile_part)
                        
                        print('      Creating quality flag for all 48 files ...')
                        calcQuality(rawdatafiles, ofileq, year, month, day, curVer, latency)
                        if os.path.isfile(ofileq_part):
                            os.remove(ofileq_part)

                    # If we don't have a complete set of obs for the day, let's still make a netcdf file ...
                    if 0 < len(rawdatafiles) < 48:
                        print('      Merging', len(rawdatafiles), 'files ...')
                        mergeGPM(rawdatafiles, ofile_part, year, month, day, var, curVer, latency)
                        print('      Creating quality flag for', len(rawdatafiles), 'files ...')
                        calcQuality(rawdatafiles, ofileq_part, year, month, day, curVer, latency)

                    if len(rawdatafiles) == 0:
                        print('      No raw data files to process')

                # Tidy up
                if (os.path.isfile(ofile)) & (os.path.isfile(ofileq)) & (len(rawdatafiles) > 0):
                    for f in rawdatafiles:
                        os.remove(f)


if __name__ == '__main__':
    latency = sys.argv[1] # Can be either 'production', 'NRTlate' or 'NRTearly'
    days = sys.argv[2] # How many days before today to process
    outdir = sys.argv[3] # Local directory
    agency = sys.argv[4] # ukmo or pagasa or bmkg or mmd
    
    try:
        end_date = datetime.datetime.strptime(sys.argv[3], "%Y%m%d").date() # Needs to be formatted YYYYMMDD
    except IndexError:
        end_date = date.today()
    main(latency, int(days), end_date, outdir, agency)
