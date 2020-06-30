import iris
import cf_units
import glob
import numpy as np
import numpy.ma as ma
import os, sys
import errno
import datetime as dt
from dateutil.relativedelta import relativedelta
import h5py
import subprocess
import location_config as config
from ftplib import FTP


'''
Downloads GPM data into an iris cube.

'''


def getTimeCoord(file):
    timeunit = cf_units.Unit('hours since 1970-01-01', calendar=cf_units.CALENDAR_GREGORIAN)

    dtstrings = os.path.basename(file).split('.')[4]

    start_dt = dt.datetime.strptime(dtstrings.split('-')[0] + '-' + dtstrings.split('-')[1], '%Y%m%d-S%H%M%S')
    end_dt = dt.datetime.strptime(dtstrings.split('-')[0] + '-' + dtstrings.split('-')[2], '%Y%m%d-E%H%M%S')
    id_pt = start_dt + ((end_dt - start_dt) / 2)

    timecoord = iris.coords.DimCoord([timeunit.date2num(id_pt)],
                                     bounds=[(timeunit.date2num(start_dt), timeunit.date2num(end_dt))],
                                     standard_name='time', units=timeunit)

    return (timecoord)


def mergeGPM(ifiles, ofile, year, month, day, var, version, latency):
    try:
        f00 = h5py.File(ifiles[0], 'r')
    except:
        print('No files found')
        return

    # Setup output cube
    geog_cs = iris.coord_systems.GeogCS(semi_major_axis=6378137, inverse_flattening=298.257223563)

    latcoord = iris.coords.DimCoord(list(f00['Grid/lat']), standard_name='latitude', units='degrees',
                                    coord_system=geog_cs)
    loncoord = iris.coords.DimCoord(list(f00['Grid/lon']), standard_name='longitude', units='degrees',
                                    coord_system=geog_cs)
    precip = np.empty((len(ifiles), 1800, 3600))

    f00.close()
    gpm_cubelist = iris.cube.CubeList([])
    for file in sorted(ifiles):
        print(file)

        try:
            f00 = h5py.File(file, 'r')
        except:
            print('There was a problem with',file)
            os.remove(file)
            continue

        filedata = list(f00['Grid/' + var])
        precip = np.dstack(filedata)
        timecoord = getTimeCoord(file)
        try:
            cube = iris.cube.Cube(precip,
                                  standard_name='precipitation_flux',
                                  units='mm h-1',
                                  attributes=None,
                                  dim_coords_and_dims=[(timecoord, 0), (latcoord, 1), (loncoord, 2)])
        except ValueError:
            cube = iris.cube.Cube(precip,
                                  standard_name='precipitation_flux',
                                  units='mm h-1',
                                  attributes=None,
                                  dim_coords_and_dims=[(timecoord, 2), (latcoord, 1), (loncoord, 0)])
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

    latcoord = iris.coords.DimCoord(list(f00['Grid/lat']), standard_name='latitude', units='degrees',
                                    coord_system=geog_cs)
    loncoord = iris.coords.DimCoord(list(f00['Grid/lon']), standard_name='longitude', units='degrees',
                                    coord_system=geog_cs)
    precip = np.empty((len(ifiles), 1800, 3600))

    f00.close()
    gpm_cubelist = iris.cube.CubeList([])
    for file in sorted(ifiles):
        # print(file)
        # pdb.set_trace()
        f00 = h5py.File(file, 'r')
        filedata = list(f00['Grid/' + var])
        precip = np.dstack(filedata)
        timecoord = getTimeCoord(file)
        try:
            cube = iris.cube.Cube(precip,
                                  long_name=field_names[var],
                                  attributes=None,
                                  dim_coords_and_dims=[(timecoord, 0), (latcoord, 1), (loncoord, 2)])
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
    return (new_cube)


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + dt.timedelta(n)


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
    return year, month, day


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
        qualindex.data[mask] = qualindex.data[mask] * ((100 - ir_weight.data[mask]) / 100.0)
        qualindex.data = ma.masked_less(qualindex.data, 0)
        qualindex.data = np.around(qualindex.data * 100).astype(np.uint8)

        qualindex.data.set_fill_value(0)

        iris.save(qualindex, ofileq, zlib=True)

    except:
        print('Error creating quality flag: ' + curVer)

def get_file_list(single_date, url):
    '''
    Get the file listing for the given url and date using curl.
    :param single_date: datetime
    :param url: server url and path to the directory
    :return: list of files (could be empty).
    '''

    cmd = 'curl -n ' + url
    args = cmd.split()
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = p.communicate()[0].decode()
    if stdout[0] == '<':
        print('No imerg files for the given date')
        return []

    filelist = stdout.split()
    filelist = [file for file in filelist if single_date.strftime("%Y%m%d-") in file]

    return filelist

def get_file(url, odir):
    '''
    Get the given file from jsimpsonhttps or arthurhouhttps using curl
    :param url: includes server, path and file to download
    :param odir: path where the file will be saved
    :return: file saved to disk, but nothing returned
    '''

    cmd = 'curl -n ' + url + ' -o ' + odir + '/' + os.path.basename(url)
    args = cmd.split()
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait() # wait so this program doesn't end before getting all files


def main(latency, start_date, end_date, agency):
    product = 'imerg'  # This shouldn't change
    # change the accounts
    settings = config.load_location_settings(agency)
    outdir = settings['gpm_path']

    server = {'production': ['https://arthurhouhttps.pps.eosdis.nasa.gov/text', settings['gpm_username'], '.HDF5'],
              'NRTlate': ['https://jsimpsonhttps.pps.eosdis.nasa.gov/text', settings['gpm_username'], '.RT-H5'],
              'NRTearly': ['https://jsimpsonhttps.pps.eosdis.nasa.gov/text', settings['gpm_username'], '.RT-H5']
              }
    var = 'precipitationCal'

    # Shouldn't need to change anything below here ..
    first_date = dt.datetime(2000, 6, 1)
    # end_date = dt.datetime.strptime(end_date, '%Y%m%d')
    # start_date = end_date - dt.timedelta(days)
    if start_date < first_date:
        start_date = first_date

    # Loop through dates
    for single_date in daterange(start_date, end_date):

        print(latency + ' : ' + single_date.strftime("%Y-%m-%d"))

        year, month, day = getYMD(single_date)
        sfilepath = {'production' : '/gpmdata/'+year+'/'+month+'/'+day+'/imerg/',
                        'NRTlate' : '/imerg/late/'+year+month+'/',
                        'NRTearly': '/imerg/early/'+year+month+'/'} # +'.*.RT-H5'

        rawdata_dir = outdir.rstrip('/') + '/rawdata/' + product + '/' + latency + '/' + year + '/' + month + '/' + day
        netcdf_dir = outdir.rstrip('/') + '/netcdf/' + product + '/' + latency + '/' + year + '/'
        ofile_test = netcdf_dir.rstrip('/') + '/gpm_' + product + '_' + latency + '_*_' + year + month + day + '.nc'
        ofile_part_test = netcdf_dir.rstrip('/') + '/gpm_' + product + '_' + latency + '_*_' + year + month + day + '_part.nc'
        ofileq_test = netcdf_dir.rstrip('/') + '/gpm_' + product + '_' + latency + '_*_' + year + month + day + '_quality.nc'
        ofileq_part_test = netcdf_dir.rstrip('/') + '/gpm_' + product + '_' + latency + '_*_' + year + month + day + '_quality_part.nc'

        if not os.path.isdir(rawdata_dir):
            mkdir_p(rawdata_dir)
        if not os.path.isdir(netcdf_dir):
            mkdir_p(netcdf_dir)

        rawdatafiles = glob.glob(rawdata_dir + '/3B-HHR*' + server[latency][2])

        if ((len(glob.glob(ofile_test)) == 1) or (len(glob.glob(ofile_part_test)) == 2)) & (
                len(glob.glob(ofileq_test)) == 1) or (len(glob.glob(ofileq_part_test)) == 2):
            print('    Nothing to do')
        else:
            attempts = 0
            filelist = get_file_list(single_date, server[latency][0] + sfilepath[latency])

            if filelist:
                print('    Downloading files from HTTPS ...')

            # Do the download
            while (len(rawdatafiles) < 48) and (attempts < 5):

                for file in filelist:
                    get_file(server[latency][0] + file, rawdata_dir)
                rawdatafiles = glob.glob(rawdata_dir + '/3B-HHR*' + server[latency][2])
                attempts += 1

            rawdatafiles = glob.glob(rawdata_dir + '/3B-HHR*' + server[latency][2])

            versions = [os.path.basename(rdf).split('.')[6] for rdf in rawdatafiles]

            for curVer in set(versions):

                print('   ', curVer)

                ofile = netcdf_dir + 'gpm_' + product + '_' + latency + '_' + curVer + '_' + year + month + day + '.nc'
                ofile_part = netcdf_dir + 'gpm_' + product + '_' + latency + '_' + curVer + '_' + year + month + day + '_part.nc'

                ofileq = netcdf_dir + 'gpm_' + product + '_' + latency + '_' + curVer + '_' + year + month + day + '_quality.nc'
                ofileq_part = netcdf_dir + 'gpm_' + product + '_' + latency + '_' + curVer + '_' + year + month + day + '_quality_part.nc'

                if os.path.isfile(ofile) & os.path.isfile(ofileq):

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


def downloadftp(rawdata_dir, server, serverpath, settings):
    '''
    The data is no longer servered by FTP. They now use HTTPS or FTPS. This functional can probably be deleted
    '''
    org = settings["organisation"]
    try:
        ftp = FTP(server[0], settings["gpm_username"], settings["gpm_username"])
        path_on_ftp = serverpath.split('3B-HHR')[0]
        file_string = serverpath.replace(path_on_ftp, '')
        ftp.cwd(path_on_ftp)
        files = ftp.nlst()
        files = [f for f in files if file_string in f]
        print(serverpath)

        for file in files:
            file_to_write = rawdata_dir + "/" + file
            if not os.path.isfile(file_to_write):
                print('Downloading: ', file_to_write)
                localfile = open(file_to_write, 'wb')
                ftp.retrbinary('RETR ' + file, localfile.write)
                localfile.close()
        ftp.quit()

    except:
        thiswd = os.getcwd()
        os.chdir(rawdata_dir)
        print('    Downloading files from FTP ...')

        downloadstring = {
            'UKMO': 'export HTTP_PROXY=http://webproxy.metoffice.gov.uk:8080 ; /opt/ukmo/utils/bin/doftp -host ' +
                    server[0] + ' -user ' + server[1] + ' -pass ' + server[1] + ' -mget ' +
                    serverpath,
            'PAGASA': 'wget --user={:s} --password={:s} {:s}:"{:s}"'.format(server[1], server[1], server[0], serverpath + '*' + server[2]),
            'BMKG': 'wget --user={:s} --password={:s} {:s}:"{:s}"'.format(server[1], server[1], server[0], serverpath + '*' + server[2]),
            'MMD': 'wget -q --user={:s} --password={:s} {:s}:"{:s}"'.format(server[1], server[1], server[0], serverpath + '*' + server[2]),
            'Andy-MacBook': 'wget -q --user={:s} --password={:s} {:s}:"{:s}"'.format(server[1], server[1], server[0], serverpath + '*' + server[2])
        }
        os.system(downloadstring[agency])
        os.chdir(thiswd)


if __name__ == '__main__':

    '''
    Usage:
    python downloadGPM.py 
                [production|NRTlate|NRTearly|auto|all]
                start_date   # Formatted as YYYYMMDD
                end_date     # Formatted as YYYYMMDD
                [PAGASA|BMKG|MMD|UKMO|Andy-MacBook]
                 
     Example:
     python downloadGPM.py auto 20191103 20191105 Andy-MacBook
    '''
    now = dt.datetime.utcnow()

    #  NB: 'auto' latency means that the most scientifically robust dataset is chosen
    latency = sys.argv[1]  # Can be either 'production', 'NRTlate' or 'NRTearly' or 'all' or 'auto'

    auto_latency = {'NRTearly': now - dt.timedelta(hours=3),
                    'NRTlate': now - dt.timedelta(hours=18),
                    'production': now - relativedelta(months=4)
                    }
    try:
        start_date = dt.datetime.strptime(sys.argv[2][:8], "%Y%m%d")  # Needs to be formatted YYYYMMDD
    except IndexError:
        start_date = now.date() - dt.timedelta(days=7)

    try:
        end_date = dt.datetime.strptime(sys.argv[3][:8], "%Y%m%d")  # Needs to be formatted YYYYMMDD
    except IndexError:
        end_date = now.date()

    agency = sys.argv[4]  # UKMO or PAGASA or BMKG or MMD or Andy-MacBook

    # Decide which latency to run the program with
    if latency == 'all':
        for l in ['production', 'NRTlate', 'NRTearly']:
            main(l, start_date, end_date, agency)
    elif latency == 'auto':
        best_latency = 'NRTearly'
        for l in auto_latency.keys():
            if end_date <= auto_latency[l]:
                best_latency = l
        main(best_latency, start_date, end_date, agency)
    else:
        main(latency, start_date, end_date, agency)
