from datetime import datetime, timedelta

'''
#######################################################################################################################
#######################################################################################################################
The functions.py module is a compilation of functions used for the download_data.py and search_missing_file.py
#######################################################################################################################
#######################################################################################################################

'''

class Functions:
	# Initialize attributes of the class Download Functions
	def __init__(self, DATA):
		self.DATA = DATA

	# get the latest available date of the desired data type depending on the input date. This function is applicable in Philippines Standard Time
	def latestDateAvailable(self, date):
		if self.DATA == 'UM':	# problem
			if date.hour < 12:	# 12 initTime
				date_yesterday = date - timedelta(days=1)
				latest_date_available = datetime(date_yesterday.year, date_yesterday.month, date_yesterday.day, 12, 0, 0)
			else:	# 00 initTime
				latest_date_available = datetime(date.year, date.month, date.day, 0, 0, 0)

		elif self.DATA == 'GSM_0.25':
			if datetime(date.year, date.month, date.day, 12, 0, 0) <= date < datetime(date.year, date.month, date.day, 18, 0, 0): # 00 initTime 
				latest_date_available = datetime(date.year, date.month, date.day, 0, 0, 0)
			elif datetime(date.year, date.month, date.day, 3, 0, 0) <= date < datetime(date.year, date.month, date.day, 6, 0, 0): # 12 initTime 
				latest_date_available = datetime(date.year, date.month, date.day, 12, 0, 0) - timedelta(days=1)
			elif datetime(date.year, date.month, date.day, 6, 0, 0) <= date < datetime(date.year, date.month, date.day, 12, 0, 0): # 18 initTime  
				latest_date_available = datetime(date.year, date.month, date.day, 18, 0, 0) - timedelta(days=1)
			elif datetime(date.year, date.month, date.day, 18, 0, 0) <= date: # 06 initTime
				if  date < datetime(date.year, date.month, date.day, 3, 0, 0):
					latest_date_available = datetime(date.year, date.month, date.day, 6, 0, 0) - timedelta(days=1)
				else:
					latest_date_available = datetime(date.year, date.month, date.day, 6, 0, 0)
			else:
				print ('Data not available yet.......')
				exit()

		elif self.DATA == 'WRF':
			if 13 <= date.hour < 16: # 00 initTime
				latest_date_available = datetime(date.year, date.month, date.day, 0, 0, 0)
			elif 16 <= date.hour < 19: # 03 initTime
				latest_date_available = datetime(date.year, date.month, date.day, 3, 0, 0)
			elif 19 <= date.hour < 22: # 06 initTime
				latest_date_available = datetime(date.year, date.month, date.day, 6, 0, 0)
			elif 22 <= date.hour or date.hour < 1:
				if date.hour < 1: # 09 initTime
					date = date - timedelta(days=1)
					latest_date_available = datetime(date.year, date.month, date.day, 9, 0, 0)
				else:
					latest_date_available = datetime(date.year, date.month, date.day, 9, 0, 0)
			elif 1 <= date.hour < 4: # 12 initTime
				date = date - timedelta(days=1)
				latest_date_available = datetime(date.year, date.month, date.day, 12, 0, 0)
			elif 4 <= date.hour < 7: # 15 initTime
				date = date - timedelta(days=1)
				latest_date_available = datetime(date.year, date.month, date.day, 15, 0, 0)
			elif 7 <= date.hour < 10: # 18 initTime
				date = date - timedelta(days=1)
				latest_date_available = datetime(date.year, date.month, date.day, 18, 0, 0)
			elif 10 <= date.hour < 13: # 21 initTime
				date = date - timedelta(days=1)
				latest_date_available = datetime(date.year, date.month, date.day, 21, 0, 0)

		elif self.DATA == 'GSMaPNRT':
			if date.minute < 40:
				date_minus12 = date - timedelta(hours=12)
				date_hour_before = date_minus12 - timedelta(hours=1)
				latest_date_available = datetime(date_hour_before.year, date_hour_before.month, date_hour_before.day, date_hour_before.hour, 40, 0)
			else:
				date_minus12 = date - timedelta(hours=12)
				latest_date_available = datetime(date_minus12.year, date_minus12.month, date_minus12.day, date_minus12.hour, 40, 0)
		
		elif self.DATA == 'GSMaPGauge':
			if date.hour < 12:
				date_3days_before = date - timedelta(days=3)
				date_yesterday = date_3days_before - timedelta(days=1)
				latest_date_available = datetime(date_yesterday.year, date_yesterday.month, date_yesterday.day, 23, 0, 0)
			else:
				date_3days_before = date - timedelta(days=3)
				latest_date_available = datetime(date_3days_before.year, date_3days_before.month, date_3days_before.day, 23, 0, 0)

		return latest_date_available

	# list all the available files on a given date and initial time (initTime) 
	def getFilesAvailable(self, initTime, date):
		if self.DATA == 'UM':
			default_dir = [date[:-2]+'T'+date[-2:]+'00Z_total_6hprecip_'+str(hour).zfill(3)+'.grib2' for hour in range(12,156,12)]

		elif self.DATA == 'GSM_0.25':
			if initTime == 12:
				for i in range(12):
					if i == 0:
						default_dir = [date+'GSM_GPV_Rra2_Gll0p25deg_Lsurf_FD'+str(i).zfill(2)+str(hour).zfill(2)+'_grib2.bin' for hour in range(0,24,3)]
					elif i == 11:
						append_dir = [date+'GSM_GPV_Rra2_Gll0p25deg_Lsurf_FD'+str(i).zfill(2)+'00_grib2.bin']
						default_dir.extend(append_dir)
					else:
						append_dir = [date+'GSM_GPV_Rra2_Gll0p25deg_Lsurf_FD'+str(i).zfill(2)+str(hour).zfill(2)+'_grib2.bin' for hour in range(0,24,3)]
						default_dir.extend(append_dir)
			else:
				for i in range(4):
					if i == 0:
						default_dir = [date+'GSM_GPV_Rra2_Gll0p25deg_Lsurf_FD0'+str(i)+str(hour).zfill(2)+'_grib2.bin' for hour in range(0,24,3)]
					elif i == 3:
						append_dir = [date+'GSM_GPV_Rra2_Gll0p25deg_Lsurf_FD0'+str(i)+str(hour).zfill(2)+'_grib2.bin' for hour in range(0,15,3)]
						default_dir.extend(append_dir)
					else:
						append_dir = [date+'GSM_GPV_Rra2_Gll0p25deg_Lsurf_FD0'+str(i)+str(hour).zfill(2)+'_grib2.bin' for hour in range(0,24,3)]
						default_dir.extend(append_dir)

		elif self.DATA == 'WRF':
			default_dir = ['pagasa_postwrf_d01_'+date[:-2]+'_'+date[-2:]+'00_f'+str(hour).zfill(3)+'00.gr2' for hour in range(0,145,1)]
			default_dir_d02 = ['pagasa_postwrf_d02_'+date[:-2]+'_'+date[-2:]+'00_f'+str(hour).zfill(3)+'00.gr2' for hour in range(0,49,1)]
			default_dir.extend(default_dir_d02)

		elif self.DATA == 'GSMaPNRT':
			default_dir = ['gsmap_nrt.'+date+'.'+str(hour).zfill(2)+'00.02_AsiaSE.csv.zip' for hour in range(initTime+1)]

		elif self.DATA == 'GSMaPGauge':
			default_dir = ['gsmap_mvk_v731120_'+date+'_'+str(hour).zfill(2)+'00_02_AsiaSE.csv.zip' for hour in range(24)]

		return default_dir