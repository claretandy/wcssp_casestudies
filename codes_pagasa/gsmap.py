import datetime
import pandas as pd
import numpy as np
import zipfile
import os

'''
#######################################################################################################################
#######################################################################################################################
The gsmap.py module extracts the hourly rain rate and computes its accumulated rainfal. There are four types of GSMaP data:
- NRT (GSMaP NRT)
- Guage_NRT (GSMaP NRT Gauge Calibrated)
- MVK (GSMaP MVK)
- Gauge (GSMaP Gauge Calibrated)
#######################################################################################################################
#######################################################################################################################

'''
class GSMaP:

	# extract accumulated rainfall within the date range depending on the selected GSMaP data type and interval (hours).
	# Returns the accumulated rainfall, latitude and longitude data.
	def extract_precipitation(self,start_date,end_date,data_type,interval):
		now_date = start_date
		colnames = ['Lat','Lon','RainRate', 'GRain']
		flag = True
		print ("CREATING DAILY PRECIPITATION (GSMaP)......................")

		while (now_date <= end_date):

			daily_precip = np.zeros([260800])

			for i in range(interval):

				if data_type in ["NRT","Gauge_NRT"]:
					directory = '/home/ict/Desktop/work/test/verification_test/data_GSMaPNRT/'+str(now_date.year).zfill(2)+str(now_date.month).zfill(2)+str(now_date.day).zfill(2)+'/'
				elif data_type == ["MVK","Gauge"]:
					directory = '/home/ict/Desktop/work/test/verification_test/data_GSMaPGauge/'+str(now_date.year).zfill(2)+str(now_date.month).zfill(2)+str(now_date.day).zfill(2)+'/'
				else:
					print ("Wrong data type.......")
					exit()

				if data_type in ["NRT","Gauge_NRT"]:
					csv = 'gsmap_nrt.'+str(now_date.year)+str(now_date.month).zfill(2)+str(now_date.day).zfill(2)+'.'\
						+str(now_date.hour).zfill(2)+'00.02_AsiaSE.csv'
				elif data_type == ["MVK","Gauge"]:
					csv = 'gsmap_mvk_v731120_'+str(now_date.year)+str(now_date.month).zfill(2)+str(now_date.day).zfill(2)+'_'\
						+str(now_date.hour).zfill(2)+'00_02_AsiaSE.csv'

				file = directory+csv

				zf = zipfile.ZipFile(file+'.zip')

				df = pd.read_csv(zf.open(csv), encoding='utf-8',header=0, names=colnames)

				if data_type in ["NRT", "MVK"]:
					precip_GSMaP = df['RainRate'].values
				elif data_type in ["Gauge_NRT","Gauge"]: 
					precip_GSMaP = df['GRain'].values

				daily_precip += precip_GSMaP

				now_date += datetime.timedelta(hours=1)

			if flag == True:
				dprecip_arr = np.array([daily_precip])
				flag = False

			else:
				dprecip_arr = np.append(dprecip_arr, np.array([daily_precip]) ,axis=0)

		latitude = df['Lat'].values
		longitude = df['Lon'].values

		return dprecip_arr, latitude, longitude
