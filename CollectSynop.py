import os
import sys
from datetime import datetime, timedelta
import location_config as config

import requests

'''
This script will collect synop observation data from BMKG CIPS, basically I want this script runs daily to collect all
synop temp and pilot data then store all the data in such a database
'''


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + dt.timedelta(n)

def getYMD(indate):
    year = indate.strftime("%Y")
    month = indate.strftime("%m")
    day = indate.strftime("%d")
    return (year, month, day)

def main(start_date, end_date, obstype, agency):
    settings = config.load_location_settings(agency)
    start_date = datetime.strptime(start_date,'%Y%m%d')
    end_date = datetime.strptime(end_date,'%Y%m%d')

    outdir = settings["datadir"]
    syn_times = ['{:02d}'.format(i) for i in range(0, 24, 3)]
    temp_times = ['{:02d}'.format(i) for i in range(0, 24, 12)]

    for single_date in daterange(start_date,end_date):
        year, month, day = getYMD(single_date)

        syn_dir = outdir + "/synop/" + year + month + day
        temp_dir = outdir + "/temp/" +year + month+ day

        if obstype == 'synop' :
            for time in syn_times :
                url = settings['db_dir']+'user='+settings['db_uname']+'&mode=web&dateRef='+year+month+date+time+'0000&timeDepth=1H&obsType=SYNOP&param=TH-PMER-TD-T-N-WIND-HU-TEND-WW_symb-VISI-CL_symb-CM_symb-CH_symb-RAF1-RAF2-TN12-TX12-TMIN10&level=0GRND&format=csv&output=binary&domain=Indonesia'
                r = requests.get(url)
                open(syn_dir+"/SYNOP"+year+month+date+time+".csv",'wb').write(r.content)
        if obstype == 'temp' :
            for time in temp_times :
                url = settings['db_dir']+'user='+settings['db_uname']+'&mode=web&dateRef='+year+month+date+time+'0000&timeDepth=1H&obsType=TEMP&param=TH-T-TD-HU-Z-WIND-TURBUL&level=1050HPA-1000HPA-950HPA-900HPA-850HPA-800HPA-700HPA-600HPA-500HPA-400HPA-300HPA-250HPA-200HPA-150HPA-100HPA-70HPA-50HPA&format=csv&output=binary&domain=Indonesia'
                r = requests.get(url)
                open(temp_dir + "/TEMP" + year + month + date + time + ".csv", 'wb').write(r.content)


if __name__ == '__main__':
    print("Downloading Synop / TEMP Observation Data")
    print("Script starts at " + str(datetime.now()))

    start_date = sys.argv[1]  # date should in YYYYMMDD
    end_date = sys.argv[2]  # start time is HH in UTC
    obstype = sys.argv[3] # synop, temp
    agency = sys.argv[4] # BMKG

    if len(date) < 8:
        print("Date should be in YYYYMMDD format")
        quit()
    if date_start > date_end:
        print("Start time must be less than end time")
        quit()

    main(date_start, date_end, obstype, agency)
    print("Finished at " + str(datetime.now()))
