import sys, os
import location_config as config
from datetime import datetime, timedelta
import pandas as pd

'''
This script will plot synop data available from BMKG synop dataset.
Basically it will look for all data in date range (downloaded previously with "CollectSynop"),
filter the data based on station ID and merge all the data in a new date indexed dataframe. The returned dataframe then
could easily plotted into timeseries graph. 
'''

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def getYMD(indate):
    year = indate.strftime("%Y")
    month = indate.strftime("%m")
    day = indate.strftime("%d")
    return year, month, day


def get_data(flist, station_id):
    #make several empty lists to contain data
    timelist = []
    tt, dewpt, mslp, wdir, wspd, cld = [], [], [], [], [], []
    rr03, rr06, rr12, rr24, = [],[],[],[]
    tx24, tn24 = [], []

    for file in flist:
        timestamp = datetime.strptime(((file.split("/"))[7])[5:15],"%Y%m%d%H")
        try:
            df_data = pd.read_csv(file,",").set_index("#id")
            timelist.append(timestamp)
            tt.append(df_data.at[station_id, "t"])
            dewpt.append(df_data.at[station_id, "td"])
            mslp.append(df_data.at[station_id, "mslp"])
            wdir.append(df_data.at[station_id, "dd"])
            wspd.append(df_data.at[station_id, "ff"])
            rr03.append(df_data.at[station_id, "rr3"])
            rr06.append(df_data.at[station_id, "rr6"])
            rr12.append(df_data.at[station_id, "rr12"])
            rr24.append(df_data.at[station_id, "rr24"])
            tn24.append(df_data.at[station_id, "tn24"])
            tx24.append(df_data.at[station_id, "tx24"])
            cld.append(df_data.at[station_id, "n"])
        except:
            tt.append("NaN")
            dewpt.append("NaN")
            mslp.append("NaN")
            wdir.append("NaN")
            wspd.append("NaN")
            rr03.append("NaN")
            rr06.append("NaN")
            rr12.append("NaN")
            rr24.append("NaN")
            tn24.append("NaN")
            tx24.append("NaN")
            cld.append("NaN")

    df_ready = pd.DataFrame({
        "time" : timelist,
        "temp" : tt,
        "dewpt" : dewpt,
        "mslp" : mslp,
        "wdir" : wdir,
        "wspd" : wspd,
        "rr03" : rr03,
        "rr06" : rr06,
        "rr12" : rr12,
        "rr24" : rr24,
        "tmin" : tn24,
        "tmax" : tx24,
        "cloud" : cld
    })

    return df_ready

def main(start_date, end_date, agency, station_id):
    settings = config.load_location_settings(agency)
    start_date = datetime.strptime(str(start_date), '%Y%m%d')
    end_date = datetime.strptime(str(end_date), '%Y%m%d')
    station_id = int(station_id)

    outdir = settings["datadir"]
    syn_times = ['{:02d}'.format(i) for i in range(0, 24, 3)]

    flist = []

    for single_date in daterange(start_date, end_date):
        year, month, day = getYMD(single_date)

        for time in syn_times:
            syn_dir = outdir + "/synop/" + year + "/" + month + "/" + day
            file = syn_dir + "/SYNOP" + year + month + day + time + ".csv"
            flist.append(file)

    df = get_data(flist, station_id)

    print(df)

if __name__ == '__main__':
    start_date = sys.argv[1]  # date should in YYYYMMDD
    end_date = sys.argv[2]  # date should in YYYYMMDD
    agency = sys.argv[3]  # BMKG
    station_id = sys.argv[4]  # BMKG

    if len(start_date) < 8 or len(end_date) < 8:
        print("Date should be in YYYYMMDD format")
        quit()
    if start_date > end_date:
        print("Start time must be less than end time")
        quit()

    main(start_date, end_date, agency, station_id)
