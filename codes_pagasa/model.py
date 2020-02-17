import pygrib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

'''
#######################################################################################################################
#######################################################################################################################
The model.py module was created for extracting meteorological values from the model depending on the input date and interval.
As of now, rainfall data extraction (accumulated rainfall) is only available.

There are four types of available model_type in this class:
- UM (Unified Model Global)
- GSM (Global Spectral Model)
- WRF_12km (WRF model 12 km)
- WRF_3km (WRF model 3 km)

Take note, there is a possibility that an error will occur if outside of the choices in the model_type was used.
#######################################################################################################################
#######################################################################################################################

'''
class Model:

	# Initialize attributes of the class Model
	def __init__(self, model_type):
		self.model_type = model_type

	''' extract information (meteorological variable, latitude, longitude, date information and date range) from the selected model
		type. Compute the accumulated rainfall by subtracting the previous date to its current date depending on the selected 
		interval (in terms of hours). UM have a minimum interval of 6 hours, GSM have 3 hour and WRF model (12km and 3km) have 1 hour. 
	'''	
	def extract(self, start_date, end_date, variable, filename, interval):
		files, dates, date_range_arr, var_index_arr, file_index_arr = self.select_files(filename, start_date, end_date, interval)
		met_var_arr = np.empty((0))

		for i in range(len(var_index_arr)):
			try:
				model_data = pygrib.open(files[int(file_index_arr[i])])
			except:
				print ('File not available ('+files[int(file_index_arr[i])]+')..........................')
				return np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), dates, date_range_arr

			if self.model_type == 'UM':
				met_var = model_data.select(name='Total water precipitation')
			elif self.model_type =='GSM':
				met_var = model_data.select(name='unknown')[0]
			elif self.model_type in ['WRF_12km', 'WRF_3km']:
				met_var = model_data.select(name='unknown', typeOfLevel='surface')[0]

			if i == 0:
				if self.model_type == 'UM':
					precip_temp = np.array([met_var[int(var_index_arr[i])].values])
				elif self.model_type == 'GSM':
					precip_temp = np.array([met_var.values[::-1]])
				else:
					precip_temp = np.array([met_var.values])

				if filename == (dates[0]-timedelta(hours=interval)).strftime("%Y%m%d%H"):	
					met_var_arr = precip_temp
				
			else:
				if self.model_type == 'UM':
					current_precip = np.array([met_var[int(var_index_arr[i])].values]) - precip_temp
					precip_temp = np.array([met_var[int(var_index_arr[i])].values])
				elif self.model_type == 'GSM':
					current_precip = np.array([met_var.values[::-1]]) - precip_temp
					precip_temp = np.array([met_var.values[::-1]])
				else:
					current_precip = np.array([met_var.values]) - precip_temp
					precip_temp = np.array([met_var.values])

				if met_var_arr.shape[0] == 0:	
					met_var_arr = current_precip
				else:
					met_var_arr = np.append(met_var_arr, current_precip, axis=0)

			model_data.close()

		if self.model_type == 'UM':
			lat_model = np.unique(met_var[0].latlons()[0])
			lon_model = np.unique(met_var[0].latlons()[1])
		else:
			lat_model = np.unique(met_var.latlons()[0])
			lon_model = np.unique(met_var.latlons()[1])

		xmesh_model, ymesh_model = np.meshgrid(lon_model, lat_model, sparse=False)

		return met_var_arr, lat_model, lon_model, dates, date_range_arr

	''' select model files that falls within the date range and it is only applicable on the single folder path.
		The filename variable has two purposes, one is the folder name and the other is the part of the 
		model filename. In this case, the folder name is in format YYYYMMDDHH (date time plus initial time)
		The var_index is applicable for UM files since a single file contains two rainfall data (6-hourly).
		While the file_index is an array containing the index for the files array.
	'''
	def select_files(self, filename, start_date, end_date, interval):

		current_date = start_date
		inital_date = datetime.strptime(filename, "%Y%m%d%H")
		files = np.empty((0))
		date_arr = np.empty((0))
		date_range_arr = np.empty((0))
		var_index = np.empty((0))
		file_index = np.empty((0))
		f_index = 0

		while current_date <= end_date:
			date_change = (current_date - inital_date).total_seconds()/3600
			date_hour = date_change

			if date_change != 0:
				if self.model_type == 'UM':
					full_path = "/home/ict/Desktop/work/test/verification_test/data_UK/"+filename+"/"
					if current_date.hour % 12 == 0:
						file = filename[:-2]+"T"+filename[-2:]+"00Z_total_6hprecip_"+str(int(date_change)).zfill(3)+".grib2"
						v_index = 1
					else:
						file = filename[:-2]+"T"+filename[-2:]+"00Z_total_6hprecip_"+str(int(date_change)+6).zfill(3)+".grib2"
						v_index = 0

				elif self.model_type == 'GSM':
					full_path = "/home/ict/Desktop/work/test/verification_test/data_GSM/0.25/"+filename+"/"
					while date_hour >= 24:
						date_hour = 0 + (date_hour%24)

					file = "GSM_GPV_Rra2_Gll0p25deg_Lsurf_FD"+str(int(date_change/24)).zfill(2)+str(int(date_hour)).zfill(2)+"_grib2.bin"

				elif self.model_type == 'WRF_12km':
					full_path = "/home/ict/Desktop/work/test/verification_test/data_WRF/"+filename+"/"
					file = 'pagasa_postwrf_d01_'+filename[:-2]+'_'+filename[-2:]+'00_f'+str(int(date_change)).zfill(3)+'00.gr2'

				elif self.model_type == 'WRF_3km':
					if date_change <= 48:
						full_path = "/home/ict/Desktop/work/test/verification_test/data_WRF/"+filename+"/"
						file = 'pagasa_postwrf_d02_'+filename[:-2]+'_'+filename[-2:]+'00_f'+str(int(date_change)).zfill(3)+'00.gr2'
					else:
						break

				try:
					files = np.append(files, full_path+file)
				except:
					print ("No such file.....")
			
			if current_date != start_date:
				date_arr = np.append(date_arr, current_date)
				string_date_range = str(current_date.hour) + 'h - ' + str((current_date-timedelta(hours=interval)).hour) + 'h'
				date_range_arr = np.append(date_range_arr, string_date_range)

			if date_change != 0:
				if self.model_type == 'UM':
					var_index = np.append(var_index, v_index)
				else:
					var_index = np.append(var_index, 0)
				file_index = np.append(file_index, f_index)
				f_index += 1

			current_date += timedelta(hours=interval)

		return files, date_arr, date_range_arr, var_index, file_index