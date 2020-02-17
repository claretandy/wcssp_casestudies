import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import functions

'''
#######################################################################################################################
#######################################################################################################################
The search_missing_file.py module look for the files (downloaded data) that are missing. The data availavle 
are UM, GSM, WRF, GSMaPNRT and GSMaPGauge.
#######################################################################################################################
#######################################################################################################################

'''

class Search_Missing:
	# Initialize attributes of the class Download Check_Download. Extract the missing file log and download log.
	def __init__(self, DATA):
		if DATA == 'UM':
			df_missing = pd.read_csv('/home/ict/Desktop/work/test/verification_test/download_files/UM_missing_data.csv',header=0)
		elif DATA == 'GSM_0.25':
			df_missing = pd.read_csv('/home/ict/Desktop/work/test/verification_test/download_files/GSM_0.25_missing_data.csv',header=0)
		elif DATA == 'WRF':
			df_missing = pd.read_csv('/home/ict/Desktop/work/test/verification_test/download_files/WRF_missing_data.csv',header=0)
		elif DATA == 'GSMaPNRT':
			df_missing = pd.read_csv('/home/ict/Desktop/work/test/verification_test/download_files/GSMaPNRT_missing_data.csv',header=0)
		elif DATA == 'GSMaPGauge':
			df_missing = pd.read_csv('/home/ict/Desktop/work/test/verification_test/download_files/GSMaPGauge_missing_data.csv',header=0)

		self.df_download_log = pd.read_csv('/home/ict/Desktop/work/test/verification_test/download_files/download_log.csv',header=0)
		self.DATA = DATA
		self.missing_data = df_missing['Missing Data'].values

	# create missing folder based on the current available folder relative to latest date available
	def createMissingFolder(self, directory, folder_name, latest_date_available):
		if self.DATA in ['UM', 'GSM_0.25', 'WRF']:
			directory_date = datetime(int(folder_name[0][0:4]), int(folder_name[0][4:6]), int(folder_name[0][6:8]), int(folder_name[0][8:10]))
		else:
			directory_date = datetime(int(folder_name[0][0:4]), int(folder_name[0][4:6]), int(folder_name[0][6:8]))

		BASE_DIR = {'UM': "/home/ict/Desktop/work/test/verification_test/data_UK/",
					'GSM_0.25': "/home/ict/Desktop/work/test/verification_test/data_GSM/0.25/",
					'WRF': "/home/ict/Desktop/work/test/verification_test/data_WRF/",
					'GSMaPNRT': "/home/ict/Desktop/work/test/verification_test/data_GSMaPNRT/",
					'GSMaPGauge': "/home/ict/Desktop/work/test/verification_test/data_GSMaPGauge/"}

		while directory_date < latest_date_available:
			if self.DATA == 'UM':
				directory_date += timedelta(hours=12)
				new_dir = directory_date
				name = "/home/ict/Desktop/work/test/verification_test/data_UK/"+str(new_dir.year).zfill(4)+\
					str(new_dir.month).zfill(2)+str(new_dir.day).zfill(2)+str(new_dir.hour).zfill(2)

			elif self.DATA == 'GSM_0.25':
				directory_date += timedelta(hours=6)
				new_dir = directory_date
				name = "/home/ict/Desktop/work/test/verification_test/data_GSM/"+self.DATA[-4:]+'/'+str(new_dir.year).zfill(4)+\
					str(new_dir.month).zfill(2)+str(new_dir.day).zfill(2)+str(new_dir.hour).zfill(2)

			elif self.DATA == 'WRF':
				directory_date += timedelta(hours=3)
				new_dir = directory_date
				name = "/home/ict/Desktop/work/test/verification_test/data_WRF/"+str(new_dir.year).zfill(4)+\
					str(new_dir.month).zfill(2)+str(new_dir.day).zfill(2)+str(new_dir.hour).zfill(2)

			elif self.DATA == 'GSMaPNRT':
				directory_date += timedelta(days=1)
				if directory_date < latest_date_available:
					new_dir = directory_date
					name = "/home/ict/Desktop/work/test/verification_test/data_GSMaPNRT/"+str(new_dir.year).zfill(4)+\
						str(new_dir.month).zfill(2)+str(new_dir.day).zfill(2)
				else:
					break

			elif self.DATA == 'GSMaPGauge':
				directory_date += timedelta(days=1)
				if directory_date < latest_date_available:
					new_dir = directory_date
					name = "/home/ict/Desktop/work/test/verification_test/data_GSMaPGauge/"+str(new_dir.year).zfill(4)+\
						str(new_dir.month).zfill(2)+str(new_dir.day).zfill(2)
				else:
					break

			print (directory_date)

			if self.DATA in ['UM', 'GSM_0.25', 'WRF']: 
				folder_name_new = str(new_dir.year).zfill(4)+str(new_dir.month).zfill(2)+str(new_dir.day).zfill(2)+str(new_dir.hour).zfill(2)
				folder_new = BASE_DIR[self.DATA] + folder_name_new
				directory = np.insert(directory, 0, folder_new)

				folder_name = np.insert(folder_name,0,str(new_dir.year).zfill(4)+str(new_dir.month).zfill(2)+str(new_dir.day).zfill(2)+str(new_dir.hour).zfill(2))
			else:
				folder_name_new = str(new_dir.year).zfill(4)+str(new_dir.month).zfill(2)+str(new_dir.day).zfill(2)
				folder_new = BASE_DIR[self.DATA] + folder_name_new
				directory = np.insert(directory, 0, folder_new)

				folder_name = np.insert(folder_name,0,str(new_dir.year).zfill(4)+str(new_dir.month).zfill(2)+str(new_dir.day).zfill(2))

			if not os.path.exists(folder_new):
				print ("Creating folder for "+self.DATA+".......................")
				os.makedirs(folder_new)

		return directory, folder_name

	# get the total available files per data type
	def filesCount(self, hour):
		if self.DATA == 'UM':
			count = 12
		elif self.DATA == 'GSM_0.25':
			if hour == 12:
				count = 89
			else:
				count = 49
		elif self.DATA == 'WRF':
			count = 194
		elif self.DATA == 'GSMaPNRT':
			count = 24
		elif self.DATA == 'GSMaPGauge':
			count = 24

		return count

	# convert the file into string format same to the format of the missing log file
	def missingFileFormat(self, file):
		if self.DATA == 'UM':
			file_format = file[0:11]+' '+file[30:33]

		elif self.DATA == 'GSM_0.25':
			file_format = file[0:10]+' '+file[42:46]

		elif self.DATA == 'WRF':
			file_format = file[19:27]+file[28:30]+' '+file[34:37]+' '+file[16:18]

		elif self.DATA == 'GSMaPNRT':
			file_format = file[10:18] + ' ' + file[19:21]

		elif self.DATA == 'GSMaPGauge':
			file_format = file[18:26] + ' ' + file[27:29]

		return file_format

	# create an array of missing files. Some missing files that are equal to the missing data stored in the log file (missing data file) are not considered
	def search_missing(self, directory, main_dir, latest_date_available):
			missing_files = np.empty(0)

			std_functions = functions.Functions(self.DATA)
			
			if self.DATA in ['UM', 'GSM_0.25', 'WRF']:
				main_dir_date = datetime(int(main_dir[0][0:4]), int(main_dir[0][4:6]), int(main_dir[0][6:8]), int(main_dir[0][8:10]))
				count_index = main_dir[0][8:10]
			else:
				main_dir_date = datetime(int(main_dir[0][0:4]), int(main_dir[0][4:6]), int(main_dir[0][6:8]))
				count_index = 0

			directory, main_dir = self.createMissingFolder(directory, main_dir, latest_date_available)

			for d in range (len(directory)):
				if len(os.listdir(directory[d])) != self.filesCount(int(count_index)):

					files = os.listdir(directory[d])
					date = main_dir[d]

					if self.DATA in ['UM', 'GSM_0.25', 'WRF']:
						latest_date_dir = datetime(int(main_dir[d][0:4]), int(main_dir[d][4:6]), int(main_dir[d][6:8]), int(main_dir[d][8:10]), 0, 0)
					else:
						if d == 0 and self.DATA == 'GSMaPNRT':
							latest_date_dir = datetime(int(main_dir[d][0:4]), int(main_dir[d][4:6]), int(main_dir[d][6:8]), latest_date_available.hour)
						else:
							latest_date_dir = datetime(int(main_dir[d][0:4]), int(main_dir[d][4:6]), int(main_dir[d][6:8]), 23)

					default_dir = std_functions.getFilesAvailable(latest_date_dir.hour, date)

					for k in default_dir:
						if not k in files and not (self.missingFileFormat(k) in self.missing_data):
							if latest_date_dir <= latest_date_available:
								missing_files = np.append(missing_files,k)

			return missing_files
