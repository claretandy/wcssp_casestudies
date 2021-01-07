import os
import sys
from datetime import datetime, timedelta
import location_config as config
import requests
import pandas as pd

'''
This script will collect synop observation data from BMKG CIPS, basically I want this script runs daily to collect all
synop temp and pilot data then store all the data in such a database
'''


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def getYMD(indate):
    year = indate.strftime("%Y")
    month = indate.strftime("%m")
    day = indate.strftime("%d")
    return year, month, day


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def mergedata(file, obstype):
    df_station = pd.read_csv("E:/DataSynop/Master.csv", sep=",", header=0)  # ,index_col="Station Number")

    if obstype == "synop":
        df_syn = pd.read_csv(file, header=0, sep=";", skiprows=[1], decimal=",")
        df_new = pd.merge(df_station, df_syn, 'inner', left_on="Station Number", right_on="#id")
        pd.DataFrame.to_csv(df_new, file, sep=",", mode="w")
    if obstype == "temp":
        df_temp = pd.read_csv(file, header=0, sep=";", skiprows=[1], decimal=",")
        df_new = pd.merge(df_station, df_temp, 'inner', left_on="Station Number", right_on="#id")
        pd.DataFrame.to_csv(df_new, file, sep=",", mode="w")


def main(start_date, end_date, obstype, agency):
    settings = config.load_location_settings(agency)
    start_date = datetime.strptime(str(start_date), '%Y%m%d')
    end_date = datetime.strptime(str(end_date), '%Y%m%d')

    outdir = settings["datadir"]
    syn_times = ['{:02d}'.format(i) for i in range(0, 24, 3)]
    temp_times = ['{:02d}'.format(i) for i in range(0, 24, 12)]

    for single_date in daterange(start_date, end_date):
        year, month, day = getYMD(single_date)

        syn_dir = outdir + "/synop/" + year +"/"+ month +"/"+ day
        temp_dir = outdir + "/temp/" + year +"/"+ month +"/"+ day

        if not os.path.isdir(syn_dir):
            mkdir_p(syn_dir)
        if not os.path.isdir(temp_dir):
            mkdir_p(temp_dir)

        if obstype == 'synop':
            for time in syn_times:
                url = settings['db_link'] + 'user=' + settings['db_uname'] + '&mode=web&dateRef=' + year + month + day + time + '0000&timeDepth=1H&obsType=SYNOP&param=TH-PMER-TD-T-N-WIND-HU-TEND-WW_symb-VISI-CL_symb-CM_symb-CH_symb-RAF1-RAF2-TN12-TX12-TMIN10&level=0GRND&format=csv&output=binary&bbox=Indonesia'
                r = requests.get(url)
                file_write = syn_dir + "/SYNOP" + year + month + day + time + ".csv"
                open(file_write, 'wb').write(r.content)
                mergedata(file_write, obstype)
        if obstype == 'temp':
            for time in temp_times:
                url = settings['db_link'] + 'user=' + settings['db_uname'] + '&mode=web&dateRef=' + year + month + day + time + '0000&timeDepth=1H&obsType=TEMP&param=TH-T-TD-HU-Z-WIND-TURBUL&level=1050HPA-1000HPA-950HPA-900HPA-850HPA-800HPA-700HPA-600HPA-500HPA-400HPA-300HPA-250HPA-200HPA-150HPA-100HPA-70HPA-50HPA&format=csv&output=binary&bbox=Indonesia'
                r = requests.get(url)
                file_write = temp_dir + "/TEMP" + year + month + day + time + ".csv"
                open(file_write, 'wb').write(r.content)
                mergedata(file_write, obstype)


if __name__ == '__main__':
    print("Downloading SYNOP / TEMP Observation Data")
    print("Script starts at " + str(datetime.now()))

    start_date = sys.argv[1]  # date should in YYYYMMDD
    end_date = sys.argv[2]  # start time is HH in UTC
    obstype = sys.argv[3]  # synop, temp
    agency = sys.argv[4]  # BMKG

    if len(start_date) < 8 or len(end_date) < 8:
        print("Date should be in YYYYMMDD format")
        quit()
    if start_date > end_date:
        print("Start time must be less than end time")
        quit()

    main(start_date, end_date, obstype, agency)
    print("Finished at " + str(datetime.now()))
