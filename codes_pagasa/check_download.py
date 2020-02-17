import pandas as pd
import numpy as np
import os

'''
#######################################################################################################################
#######################################################################################################################
The check_download.py module determines if the input files were downloaded completely and updates the log file according
to its latest date available. This module is mainly used in the download_data.py module.
#######################################################################################################################
#######################################################################################################################

'''

class Check_Download:
	# Initialize attributes of the class Download Check_Download
	def __init__(self, DATA):
		if DATA in ['UM', 'GSM_0.25', 'WRF', 'GSMaPNRT', 'GSMaPGauge']:
			self.DATA = DATA
			if DATA == 'GSM_0.25':
				self.BASE_DIR = '/home/ict/Desktop/work/test/verification_test/data_'+DATA[0:3]+'/0.25/'
			else:
				self.BASE_DIR = '/home/ict/Desktop/work/test/verification_test/data_'+DATA+'/'
		else:
			print ("Wrong input of Data type...........")

	# extract information (directory, filename, missing file and index) from the input file
	def getFileInformation(self, file):
		if self.DATA in ['UM', 'GSM_0.25', 'WRF']:
			if self.DATA == 'UM':
				initTime = file[9:11]
				filename = file
				hour_file = filename[30:33]
				directory = file[0:8] + initTime
				missing_file = directory+' '+hour_file
				index = 3
			elif self.DATA == 'GSM_0.25':
				initTime = file[8:10]
				filename = file[10:]
				hour_file = file[42:46]
				directory = file[0:10]
				missing_file = directory+' '+hour_file
				index = 2

			elif self.DATA == 'WRF':
				initTime = file[28:30]
				filename = file
				hour_file = filename[34:37]
				directory = file[19:27] + file[28:30]
				missing_file = directory+' '+hour_file+' '+filename[16:18]
				index = 4

			return initTime, filename, hour_file, directory, missing_file, index
		else:	
			if self.DATA == 'GSMaPNRT':
				directory = file[10:18]
				filename = file
				missing_file = directory+' '+filename[19:21]
				index = 0
			elif self.DATA == 'GSMaPGauge':
				directory = file[18:26]
				filename = file
				missing_file = directory+' '+filename[27:29]
				index = 1

			return directory, filename, missing_file, index

	''' loop to individual files if it is downloaded completely. If yes, update the log file to its recent date.
		If no, log file will not be updated and put that file to pending files log file. If the the number of attempt in pending files
		exceeds, it will be transferred on the missing file log.
	'''
	def check_download(self, files, date, date_start):
		pending_flag = False

		for f in range(len(files)):
			self.df_download_log = pd.read_csv('/home/ict/Desktop/work/test/verification_test/download_files/download_log.csv',header=0)
			self.df_missing = pd.read_csv('/home/ict/Desktop/work/test/verification_test/download_files/'+self.DATA+'_missing_data.csv',header=0)
			self.df_pending = pd.read_csv('/home/ict/Desktop/work/test/verification_test/download_files/'+self.DATA+'_pending_data.csv',header=0)

			if self.DATA in ['UM', 'GSM_0.25', 'WRF']:
				initTime, filename, hour_file, directory, missing_file, data_index = self.getFileInformation(files[f])
			else:
				directory, filename, missing_file, data_index = self.getFileInformation(files[f])

			if os.path.isfile(self.BASE_DIR+directory+'/'+filename) and pending_flag==False:
				if self.DATA in ['UM', 'GSM_0.25', 'WRF']:
					date_start = str(date.year).zfill(4)+str(date.month).zfill(2)+str(date.day).zfill(2) + initTime
					self.df_download_log.iat[data_index,2] = int(hour_file)
				else:
					date_start = str(date.year).zfill(4)+str(date.month).zfill(2)+str(date.day).zfill(2)
					self.df_download_log.iat[data_index,2] = 0

				self.df_download_log.iat[data_index,1] = int(date_start)
				self.df_download_log.to_csv('/home/ict/Desktop/work/test/verification_test/download_files/download_log.csv', index=False)

			else:
				pending_files = self.df_pending['Pending Data'].values
				pending_attempt = self.df_pending['Attempt'].values

				if missing_file in pending_files:
					index = np.where(pending_files==missing_file)[0][0]
					attempt = pending_attempt[index]

					if attempt == 4:
						df_add = {'Missing Data': missing_file}
						self.df_missing = self.df_missing.append(df_add, ignore_index=True)
						self.df_missing.to_csv('/home/ict/Desktop/work/test/verification_test/download_files/'+self.DATA+'_missing_data.csv', index=False)

						self.df_pending.drop([index])
						self.df_pending.drop([index]).to_csv('/home/ict/Desktop/work/test/verification_test/download_files/'+self.DATA+'_pending_data.csv', index=False)

						if pending_flag == False:
							if self.DATA in ['UM', 'GSM_0.25', 'WRF']:
								date_start = str(date.year).zfill(4)+str(date.month).zfill(2)+str(date.day).zfill(2) + initTime
							else:
								date_start = str(date.year).zfill(4)+str(date.month).zfill(2)+str(date.day).zfill(2)

							self.df_download_log.iat[data_index,1] = int(date_start)
							self.df_download_log.iat[data_index,2] = 0
							self.df_download_log.to_csv('/home/ict/Desktop/work/test/verification_test/download_files/download_log.csv', index=False)

					elif attempt < 4:
						date_start = directory
						self.df_download_log.iat[data_index,1] = int(date_start)
						self.df_download_log.to_csv('/home/ict/Desktop/work/test/verification_test/download_files/download_log.csv', index=False)

						self.df_pending.iat[index,1] = attempt + 1
						self.df_pending.to_csv('/home/ict/Desktop/work/test/verification_test/download_files/'+self.DATA+'_pending_data.csv', index=False)

						pending_flag = True

				else:
					df_add = {'Pending Data': missing_file,
								'Attempt': 1}
					self.df_pending = self.df_pending.append(df_add, ignore_index=True)
					self.df_pending.to_csv('/home/ict/Desktop/work/test/verification_test/download_files/'+self.DATA+'_pending_data.csv', index=False)

					pending_flag = True