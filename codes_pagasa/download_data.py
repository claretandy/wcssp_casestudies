from datetime import datetime, timedelta
import os
import ftplib
import numpy as np
import time
import pandas as pd
import search_missing_file
import check_download
import functions
import sys
from urllib.error import HTTPError, URLError
from socket import timeout
import urllib.request
import requests

'''
#######################################################################################################################
#######################################################################################################################
The download_data.py module downloads the lastest available data on the current date and time. The module can 
fillup the missing files and downloads (update function) it. In addition, the update will start to the latest 
completed date which is stored in a log file (csv file). After the download or update, it will check if the files were
downloaded completely and updates the log file. The available data for download are:

- UM (Unified Model Global)
- GSM_0.25 (Global Spectral Model 0.25 degrees)
- WRF (Weather Research and Forecasting Model)
- GSMaPNRT (GSMaP NRT version txt file)
- GSMaPGauge (GSMaP Gauge version txt file)
#######################################################################################################################
#######################################################################################################################
'''

class Download_Data:
	''' Initialize attributes of the class Download Data
		extract start date from the log file
	'''
	def __init__(self, data_type):
		colnames = ['Data','Date','Hour']
		df=pd.read_csv('/home/ict/Desktop/work/test/verification_test/download_files/download_log.csv',header=0, names=colnames)

		if data_type == 'WRF':
			self.date_start = str(df['Date'].values[4])
			self.hour_start = str(df['Hour'].values[4])

		elif data_type == 'UM':
			self.date_start = str(df['Date'].values[3])
			self.hour_start = str(df['Hour'].values[3])

		elif data_type == 'GSM_0.25':
			self.date_start = str(df['Date'].values[2])
			self.hour_start = str(df['Hour'].values[2])

		elif data_type == 'GSMaPNRT':
			self.date_start = str(df['Date'].values[0])
			self.hour_start = str(df['Hour'].values[0])

		elif data_type == 'GSMaPGauge':
			self.date_start = str(df['Date'].values[1])
			self.hour_start = str(df['Hour'].values[1])

		self.data_type = data_type

	''' download the file from the ftp server provided by username and password. It will
		try to download the file within 10 seconds and will stop downloading if the desired
		file does not exist on the ftp + filepath
	'''
	def download(self, ftp, username, password, filepath, filename, data_type):

		ftp = ftplib.FTP(ftp, timeout=10) 
		ftp.login(username, password) 

		if data_type != 'GSM_0.25':
			if filepath != 0:
				if not ftp.nlst(filepath) == []:
					ftp.cwd(filepath)
					if filename in ftp.nlst():
						print ("Downloading......................", filename, filepath)
						ftp.retrbinary("RETR " + filename, open(filename, 'wb').write)
					else:
						print ("File not exist in the ftp...............", filename, filepath)

				else:
					print ("FTP Directory not exist...................", filename, filepath)

			else:
				if filename in ftp.nlst():
					print ("Downloading......................", filename, filepath)
					ftp.retrbinary("RETR " + filename, open(filename, 'wb').write)
				else:
					print ("File not exist in the ftp...............", filename, filepath)

			ftp.quit()

		else:
			password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
			base_url = "https://www.wis-jma.go.jp/d/c/RJTD/GRIB/Global_Spectral_Model/Latitude_Longitude/0.25_0.25/90.0_-5.0_30.0_195.0/Surface_layers/"

			password_mgr.add_password(None, base_url, 'wis_jma0018ph', '7K5j7yoTr3g8')

			handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
			opener = urllib.request.build_opener(handler)

			urllib.request.install_opener(opener)
			r = requests.head(base_url + filepath[:-2] +"/"+filepath[-2:]+"0000/"+filename[10:], auth=('wis_jma0018ph', '7K5j7yoTr3g8'))
	
			if r.status_code == 200:
				response = urllib.request.urlopen(base_url + filepath[:-2] +"/"+filepath[-2:]+"0000/"+filename[10:], timeout=10)
				print ("Downloading......................", filename[10:], filename[0:10])
				with open(filename[10:], 'wb') as f:

				    try:
				        f.write(response.read())
				    except:
				        print("error")
				        os.system('rm '+filename)

			else:
				print ("File not exists in the server...............", filename[10:], filename[0:10])

	# return the base folder where the data was stored
	def get_base_folder(self, data_type):
		if data_type == 'WRF':
			base_folder = '/home/ict/Desktop/work/test/verification_test/data_WRF/'

		elif data_type == 'UM':
			base_folder = '/home/ict/Desktop/work/test/verification_test/data_UK/'

		elif data_type == 'GSM_0.25':
			base_folder = '/home/ict/Desktop/work/test/verification_test/data_GSM/0.25/'

		elif data_type == 'GSMaPNRT':
			base_folder = '/home/ict/Desktop/work/test/verification_test/data_GSMaPNRT/'

		elif data_type == 'GSMaPGauge':
			base_folder = '/home/ict/Desktop/work/test/verification_test/data_GSMaPGauge/'

		return base_folder

	# convert the file data to string name of the directory
	def get_file_directory(self, data_type, file):
		if data_type == 'WRF':
			file_dir = file[19:27] + file[28:30]

		elif data_type == 'UM':
			file_dir = file[0:8] + file[9:11]

		elif data_type == 'GSM_0.25':
			file_dir = file[0:10]

		elif data_type == 'GSMaPNRT':
			file_dir = file[10:18]

		elif data_type == 'GSMaPGauge':
			file_dir = file[18:26]

		return file_dir

	# return the file path of each data type
	def get_file_path(self, data_type, file):
		if data_type == 'WRF':
			file_path = "/ihpc/wrf/HMD/" + file[19:27] + "_" + file[28:30] + "00"

		elif data_type == 'UM':
			file_path = "/tamss/hmd/Unified_Model/0000/"

		elif data_type == 'GSM_0.25':
			file_path = file

		elif data_type == 'GSMaPNRT':
			file_path = "/realtime_ver/v7/txt/02_AsiaSE/"+file[10:14]+'/'+file[14:16]+'/'+file[16:18]

		elif data_type == 'GSMaPGauge':
			file_path = "/standard/v7/txt/hourly/02_AsiaSE/"+file[18:22]+'/'+file[22:24]+'/'+file[24:26]

		return file_path

	# search for missing file data and download it
	def update_Data(self, missing_flag, files):
		print ("Updating " + self.data_type + ".......................")
		std_functions = functions.Functions(self.data_type)
		latest_date_available = std_functions.latestDateAvailable(datetime.now())
		base_folder = self.get_base_folder(self.data_type)

		if not missing_flag:
			if not os.path.exists(base_folder+self.date_start):
				os.makedirs(base_folder+self.date_start)
				
			sorted_dir = np.asarray(sorted(os.listdir(base_folder), reverse=True))

			index = np.where(sorted_dir == self.date_start)[0][0]
			daily_data = [os.path.join(base_folder,s) for s in sorted_dir[0:index+1]]

			smf = search_missing_file.Search_Missing(self.data_type)

			missing_files = smf.search_missing(daily_data, sorted_dir[0:index+1], latest_date_available)

		else:
			missing_files = files

		for file in missing_files:

			directory = self.get_file_directory(self.data_type, file)

			if not os.path.exists(base_folder+directory):
				os.makedirs(base_folder+directory)
			
			if not os.path.isfile(base_folder+directory+'/'+file):

				if self.data_type == 'UM':
					filename = file + '.gz'
				else:
					filename = file

				time_flag = 1

				while True:
		   			try:
		   				old_dir=os.getcwd()
		   				os.chdir(base_folder+directory)

		   				if self.data_type != 'GSM_0.25':
		   					filepath = self.get_file_path(self.data_type, file)

		   				if self.data_type == 'UM':
		   					self.download(ftp, username, password, 0, file, self.data_type)	# insert ftp, username and password
		   					os.system('gunzip '+filename)

		   				elif self.data_type == 'GSM_0.25':
		   					self.download(ftp, username, password, directory, filename, self.data_type)	# insert ftp, username and password

		   				else:
			   				self.download(ftp, username, password, filepath, file, self.data_type)	# insert ftp, username and password

		   				os.chdir(old_dir)

		   				break

		   			except ftplib.all_errors:
		   				os.chdir(old_dir)
		   				print ("No internet connection......................", time_flag)

		   				if os.path.isfile(base_folder+directory+'/'+file):

		   					os.chdir(base_folder+directory)
		   					os.system('rm '+file)
		   					os.chdir(old_dir)

		   				time.sleep(10.0 - ((time.time() %  10.0)))

		   				if time_flag == 2:
		   					break

		   				time_flag += 1
		   				pass

		   			except (HTTPError, URLError) as error:
		   				os.chdir(old_dir)
		   				print ("No internet connection......................", time_flag)

		   				if os.path.isfile(base_folder+directory+'/'+filename):
		   					os.chdir(base_folder+directory)
		   					os.system('rm '+filename)
		   					os.chdir(old_dir)

		   				time.sleep(10.0 - ((time.time() %  10.0)))

		   				if time_flag == 2:
		   					break

		   				time_flag += 1
		   				pass

		   			except timeout:
		   				os.chdir(old_dir)
		   				print ("No internet connection......................", time_flag)

		   				if os.path.isfile(base_folder+directory+'/'+filename):
		   					os.chdir(base_folder+directory)
		   					os.system('rm '+filename)
		   					os.chdir(old_dir)

		   				time.sleep(10.0 - ((time.time() %  10.0)))

		   				if time_flag == 2:
		   					break

		   				time_flag += 1
		   				pass

		if len(missing_files) != 0:
			cd = check_download.Check_Download(self.data_type)
			cd.check_download(missing_files, latest_date_available, self.date_start)

		print (self.data_type+" Updated..............................")	
	
	# download the recent available data from the current datetime. if the download is not complete, proceed to update_data function
	def download_Data(self):
		print (self.data_type + ".......................download")

		std_functions = functions.Functions(self.data_type)
		latest_date_available = std_functions.latestDateAvailable(datetime.now())
		base_folder = self.get_base_folder(self.data_type)

		if self.data_type in ['GSMaPNRT', 'GSMaPGauge']:
			directory = latest_date_available.strftime("%Y%m%d")
		else:
			directory = latest_date_available.strftime("%Y%m%d%H")

		if self.data_type == 'GSMaPNRT':
			file_arr = ['gsmap_nrt.'+latest_date_available.strftime("%Y%m%d")+'.'+latest_date_available.strftime("%H")+'00.02_AsiaSE.csv.zip']
		else:
			file_arr = std_functions.getFilesAvailable(latest_date_available.hour, directory)
			
		if not os.path.exists(base_folder+directory):
		    os.makedirs(base_folder+directory)

		time_flag = 1

		for file in file_arr:
			if not os.path.isfile(base_folder+directory+'/'+file) or os.path.isfile(base_folder+directory+'/'+file[10:]):

				if self.data_type == 'UM':
					filename = file + '.gz'

				elif self.data_type == 'GSM':
					filename = file[10:]
				else:
					filename = file

				while True:
					try:
						old_dir=os.getcwd()
						os.chdir(base_folder+directory)

						if self.data_type != 'GSM_0.25':
		   					filepath = self.get_file_path(self.data_type, file)

						if self.data_type == 'UM':
		   					self.download(ftp, username, password, 0, file, self.data_type)	# insert ftp, username and password
		   					os.system('gunzip '+filename)

		   				elif self.data_type == 'GSM_0.25':
		   					self.download(ftp, username, password, directory, filename, self.data_type)	# insert ftp, username and password

		   				else:
			   				self.download(ftp, username, password, filepath, file, self.data_type)	# insert ftp, username and password

						os.chdir(old_dir)

						break

					except ftplib.all_errors:
						os.chdir(old_dir)
						print ("No internet connection......................")

						if os.path.isfile(base_folder+directory+'/'+file):

							os.chdir(base_folder+directory)
							os.system('rm '+file)
							os.chdir(old_dir)

						time.sleep(10.0 - ((time.time() %  10.0)))
						if time_flag == 2:
							break
						time_flag += 1
						
						pass

					except (HTTPError, URLError) as error:
						os.chdir(old_dir)
						print ("No internet connection......................", time_flag)

						if os.path.isfile(base_folder+directory+'/'+filename):
							os.chdir(base_folder+directory)
							os.system('rm '+filename)
							os.chdir(old_dir)

						time.sleep(10.0 - ((time.time() %  10.0)))
						if time_flag == 2:
							break
						time_flag += 1
						pass

					except timeout:
						os.chdir(old_dir)
						print ("No internet connection......................", time_flag)

						if os.path.isfile(base_folder+directory+'/'+filename):

							os.chdir(base_folder+directory)
							os.system('rm '+filename)
							os.chdir(old_dir)

						time.sleep(10.0 - ((time.time() %  10.0)))
						if time_flag == 2:
							break
						time_flag += 1
						pass

		sorted_dir = np.asarray(sorted(os.listdir(base_folder), reverse=True))

		if not os.path.exists(base_folder+self.date_start):
		    os.makedirs(base_folder+self.date_start)

		index = np.where(sorted_dir == self.date_start)[0][0]
		daily_Data = [os.path.join(base_folder,s) for s in sorted_dir[1:index+1]]

		if len(sorted_dir[1:index+1]) != 0:
			smf = search_missing_file.Search_Missing(self.data_type)

			missing_files = smf.search_missing(daily_Data, sorted_dir[1:index+1], latest_date_available)

			if len(missing_files) != 0:
				self.update_Data(True, missing_files)
			else:
				cd = check_download.Check_Download(self.data_type)
				cd.check_download(missing_files, latest_date_available, self.date_start)

		else:
			cd = check_download.Check_Download(self.data_type)
			cd.check_download(file_arr, latest_date_available, self.date_start)

if __name__ == '__main__':
	try:
		data_type = sys.argv[1]
		download_type = sys.argv[2]
	except:
		print ("Choose data type (UM, WRF, GSM_0.25, GSMaPNRT, GSMaPGauge) and download type (download or update only)................")

	download = Download_Data(data_type)

	if download_type == "download":
		download.download_Data()
	elif download_type == "update":
		download.update_Data(False, None)
	else:
		print ("Wrong input of download type................... (download or update only)")