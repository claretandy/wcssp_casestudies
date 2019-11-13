import os
from datetime import datetime
from netCDF4 import Dataset
from math import radians, sin, cos, atan2, sqrt, log, fabs, degrees, atan
import numpy as np


def dist_haversine(lat1, lon1, lat2, lon2, R = 6371.0):
    """
    Returns the great-circle distance (in km) between two points (given in lat-lon),
    assuming Earth is a sphere with radius R (default = 6371.0 km) 
    """
    lat1 = radians(lat1); lon1 = radians(lon1)
    lat2 = radians(lat2); lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = pow(sin(0.5*dlat), 2) + cos(lat1)*cos(lat2)*pow(sin(0.5*dlon), 2)
    c = 2*atan2(sqrt(a), sqrt(1-a))
    d = R*c

    return d

def isinside_R(lat, lon, lat0, lon0, R):
    """
    Returns TRUE if point (lat,lon) is inside circle with center (lat0, lon0) and radius R
    """
    rad = dist_haversine(lat0, lon0, lat, lon)
    if rad > R:
        return False
    else:
        return True
v_isinside_R = np.vectorize(isinside_R)

def get_variables(ddir, dtime):
    """
    Gets numpy array of values from netcdf files stored in ddir for dtime
    
    Returns dictionaries temp (rh and temp) and wind (u and v) containing:
        pres        1D-np.array, pressure levels
        lon         1D-np.array, longitude
        lat         1D-np.array, latitude
        rh/temp/u/v 3D-np.array of values
    """
    var = {'rh':'rh_wrt_water_levels-inst', 'temp':'templevels-inst', 'u':'Uwind-inst', 'v':'Vwind-inst'}
    ncvar = {'rh':'relative_humidity', 'temp':'air_temperature', 'u':'x_wind', 'v':'y_wind'}

    temp = {}; wind = {}
    for k,v in var.items():
        fname = '{:s}/{:s}_analysis_{:s}.nc'.format(ddir, dtime.strftime('%Y%m%dT%H%MZ'), v)
        assert os.path.exists(fname), fname+' does not exist!'
        
        if k in ['u', 'v']:
            ncfile = Dataset(fname)
            if k == 'u':
                wind['pres'] = ncfile['pressure'][:].data
                wind['lon'] = ncfile['longitude'][:].data
                wind['lat'] = ncfile['latitude'][:].data
            wind[k] = ncfile[ncvar[k]][:].data
            ncfile.close()
        elif k in ['rh', 'temp']:
            ncfile = Dataset(fname)
            if k == 'rh':
                temp['pres'] = ncfile['pressure'][:].data
                temp['lon'] = ncfile['longitude'][:].data
                temp['lat'] = ncfile['latitude'][:].data
                temp[k] = ncfile[ncvar[k]][:].data
            elif k == 'temp':
                temp[k] = ncfile[ncvar[k]][:].data
            ncfile.close()
    
    return temp, wind

def compute_td(RH, t):
    """
    Computes td given RH and t bases on Eq. 8 in Lawrence (2004).
    Coefficients are as stated in text, based on Alduchov and Eskridge (1986)
    
    Lawrence, M.G., 2004: The Relationship between Relative Humidity and the Dewpoint
    Temperature in Moist Air - A Simple Conversion and Applications. DOI:10.1175/BAMS-86-2-225
    """
    A1 = 7.625
    B1 = 243.04 # deg C
    
    try:
        num = B1*(log(RH/100) + A1*t/(B1 + t))
        den = A1 - log(RH/100) - A1*t/(B1 + t)
    except:
        return np.nan
    
    return num/den
v_compute_td = np.vectorize(compute_td)

def compute_dir(u, v):
    """
    Computes direction of wind (with North as 0-deg)
    Note that if u and v are both positive, wind is southwesterly, direction is 90 - atan(v, u)
    """
    if all([u == 0, v == 0]):
        return 0   # wind at rest
    elif all([u == 0, v > 0]):
        return 180 # southerly
    elif all([u == 0, v < 0]):
        return 0   # northerly
    elif all([u > 0, v == 0]):
        return 270  # westerly
    elif all([u < 0, v == 0]):
        return 90 # easterly
    
    theta = fabs(degrees(atan(v/u)))
    if all([u > 0, v > 0]):
        return 270 - theta
    elif all([u > 0, v < 0]):
        return 270 + theta
    elif all([u < 0, v < 0]):
        return 90 - theta
    else:
        return 90 + theta
v_compute_dir = np.vectorize(compute_dir)

def get_sounding_temp(temp, lat0, lon0, radius=20):
    """
    Returns 2d array of pres, temp, and dwpt at point(lat0, lon0), where
    temp and dwpt are averaged over a radius-km from point (in deg Celsius)
    
    temp    dictionary output from get_variables
    """   
    # Define grid of lats and lons
    tlons, tlats = np.meshgrid(temp['lon'], temp['lat'])
    # Create mask inside specified radius
    mask = v_isinside_R(tlats, tlons, lat0, lon0, radius)
    # Repeat mask to 3D
    masks = np.repeat(mask[None,...], len(temp['pres']), axis=0)
    # Create masked array to average; convert temp to C
    temp_masked = np.ma.array(temp['temp'] - 273.16, mask=~masks, fill_value=np.nan)
    rh_masked = np.ma.array(temp['rh'], mask=~masks, fill_value=np.nan)
    
    # Get average
    temp_pt = np.nanmean(np.nanmean(temp_masked, axis=1), axis=1).data
    rh_pt = np.nanmean(np.nanmean(rh_masked, axis=1), axis=1).data
    
    # Compute Td based on averaged values
    dwpt_pt = v_compute_td(rh_pt, temp_pt)
    
    return np.vstack((temp['pres'], temp_pt, dwpt_pt)).T

def get_sounding_wind(wind, lat0, lon0, radius=20):
    """
    Returns 2d array of pres, u, v, wspd, and wdir at point(lat0, lon0), where
    u and v are averaged over a radius-km from point (in knots)
    
    wind    dictionary output from get_variables
    """   
    # Define grid of lats and lons
    wlons, wlats = np.meshgrid(wind['lon'], wind['lat'])
    # Create mask inside specified radius
    mask = v_isinside_R(wlats, wlons, lat0, lon0, radius)
    # Repeat mask to 3D
    masks = np.repeat(mask[None,...], len(wind['pres']), axis=0)
    # Create masked array to average; convert wind to knots
    u_masked = np.ma.array(wind['u'] * 1.943844, mask=~masks, fill_value=np.nan)
    v_masked = np.ma.array(wind['v'] * 1.943844, mask=~masks, fill_value=np.nan)
    
    # Get average
    u_pt = np.nanmean(np.nanmean(u_masked, axis=1), axis=1).data
    v_pt = np.nanmean(np.nanmean(v_masked, axis=1), axis=1).data
    wspd = np.sqrt(np.square(u_pt) + np.square(v_pt))
    wdir = v_compute_dir(u_pt, v_pt)
    
    return np.vstack((wind['pres'], u_pt, v_pt, wspd, wdir)).T
    

dtime = datetime(2019,11,1,12)

ddir = './data/um'

temp, wind = get_variables(ddir, dtime)

lat0 = 14.5812
lon0 = 121.3693

tanay_temp = get_sounding_temp(temp, lat0, lon0)
tanay_wind = get_sounding_wind(wind, lat0, lon0)

