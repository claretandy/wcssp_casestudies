import numpy as np
from scipy import spatial

'''
#######################################################################################################################
#######################################################################################################################
The idw.py module interpolate the output data using the Inverse Distance Weighted (IDW) Interpolation.
#######################################################################################################################
#######################################################################################################################

'''

class IDW:
	# Calculate the great circle distance between two point on the earth (specified in decimal degrees)
	# All args must be of equal length. 
	def haversine_np(self, lon1, lat1, lon2, lat2):
	    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

	    dlon = lon2 - lon1
	    dlat = lat2 - lat1

	    a = np.sin(dlat/2.0)**2.0 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2.0

	    c = 2.0 * np.arcsin(np.sqrt(a))
	    km = 6367.0 * c
	    return km

	# interpolate z data to the new latitude and longitude data (x_new and y_new). 
	# The function uses a KD tree to find eight nearest neighbor from the reference point. 
	def interpolate(self,x_new,y_new,x,y,z,power):
		print ('INTERPOLATING.....................................')
		coor = np.vstack((y,x)).T
		coordinates = np.array([y,x])
		idw_arr = np.array([])
		x_arr = np.vstack(([x_new]*8)).T
		y_arr = np.vstack(([y_new]*8)).T
		coor = np.vstack((y, x)).T
		coor_new = np.vstack((y_new, x_new)).T
		tree = spatial.KDTree(coor)

		close_dist, close_arr = tree.query([coor_new],k=8)

		points_old = coor[close_arr,:]

		distance = self.haversine_np(x_arr, y_arr, points_old[0,:,:,1], points_old[0,:,:,0])

		value = z[:, close_arr[0,:,:]]

		num_temp = value/(distance**power)

		num_temp[np.where(np.isnan(num_temp)==True)] = 0
		num_temp[np.where(np.isinf(num_temp)==True)] = 0

		den_temp = 1/(distance**power)
		den_temp[np.where(np.isnan(den_temp)==True)] = 0
		den_temp[np.where(np.isinf(den_temp)==True)] = 0

		numerator = np.sum(num_temp, axis=2)
		denominator = np.sum(den_temp, axis=1)

		idw_arr = numerator/denominator

		return idw_arr