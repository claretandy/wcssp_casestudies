import sys, os
import location_config as config
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pandas.plotting import register_matplotlib_converters
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
register_matplotlib_converters()

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

def plotdata(df, station_id, start_date, end_date):
    #get station info
    print(df)
    df.fillna(method="ffill")
    df_station = pd.read_csv("E:/DataSynop/Master.csv", sep=",", header=0 ,index_col="Station Number")
    st_name = df_station.at[station_id,"Station Name"]
    st_lat = df_station.at[station_id,"latitude"]
    st_lon = df_station.at[station_id,"longitude"]
    st_elev = df_station.at[station_id,"Elevation"]

    myfmt = mdates.DateFormatter("%d/%m %HZ")
    myfmt2 = mdates.DateFormatter("%d/%H")
    minfmt = mdates.DateFormatter("%H")

    #set plot
    fig = plt.figure(figsize=(10,7))
    gs = gridspec.GridSpec(4,4)

    ax0 = fig.add_subplot(gs[0:2,:])
    ax0.plot(df["time"],df["temp"],linewidth=1.2, linestyle='-', marker='o', color='#0B2E59', label='Temperature')
    ax0.plot(df["time"],df["dewpt"],linewidth=1.2, linestyle='-', marker='o', color='#7AB317', label='Dew Point')
    ax0.plot(df["time"],df["tmax"], '2r', label='T Max')
    ax0.plot(df["time"],df["tmin"], '1b', label='T Min')
    ax0.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax0.xaxis.set_major_formatter(myfmt)
    ax0.xaxis.set_minor_locator(ticker.AutoMinorLocator(8))
    ax0.tick_params(labelsize=8)
    #ax0.set_ylim(20,37)
    ax0.grid(True)
    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)
    fig.suptitle('Synop Observation Data at ' + str(st_name) +' ('+str(station_id)+')\n'
                 +str(st_lat)+"N "+str(st_lon)+"E "+ str(st_elev)+ "m \n"
                 +"from "+ str(start_date)+ " to "+str(end_date))

    ax1 = fig.add_subplot(gs[2,0:2])
    ax1.plot(df["time"],df["rr03"], label='3hr Rainfall')
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(myfmt)
    ax1.tick_params(labelsize=8)
    ax1.legend(prop={'size': 7})
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(8))

    ax2 = fig.add_subplot(gs[2,2:])
    ax2.plot(df["time"],df["rr24"], label='24hr Rainfall')
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(myfmt)
    ax2.tick_params(labelsize=8)
    ax2.legend(prop={'size': 7})
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(8))

    ax3 = fig.add_subplot(gs[3,0:2])
    ax3.plot(df["time"], df["mslp"], label='MSLP')
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax3.xaxis.set_major_formatter(myfmt2)
    ax3.legend(prop={'size': 6})
    ax3.tick_params(labelsize=7)

    ax4 = fig.add_subplot(gs[3,2])
    ax4.plot(df["time"], df["wspd"], label="FF")
    ax4.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax4.tick_params(labelsize=7, which='major')
    ax4.legend(prop={'size': 6})
    ax4.xaxis.set_major_formatter(myfmt2)

    ax5 = fig.add_subplot(gs[3,3])
    ax5.plot(df["time"], df["cloud"], label="Cloud")
    ax5.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax5.tick_params(labelsize=7, which='major')
    ax5.legend(prop={'size': 6})
    ax5.xaxis.set_major_formatter(myfmt2)

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

    plotdata(df, station_id, start_date, end_date)

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
