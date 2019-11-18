import ftplib
import datetime
from datetime import timedelta
import fnmatch
import shutil
import os
import gzip


today = datetime.date.today()
yesterday = today - timedelta(days=1)
first = today - timedelta(days=7)
year=str(yesterday)[0:4]
month=str(yesterday)[5:7]
last=str(yesterday)[8:10]

path = '/realtime_ver/v7/hourly_G/'+year+'/'+month+'/'+last+'/'

ftp = ftplib.FTP("133.56.96.215","rainmap","Niskur+1404")
ftp.cwd(path)
dataloc='I:/Nanda/NDF/GSMaP/'
flist=ftp.nlst()

skipped = 0
for file in flist:
    proses=ftp.retrbinary("RETR " + file ,open(dataloc+file, 'wb').write)
    print ('Downloading :'+str(proses)+' File:'+str(file))
ftp.quit()

for f,a in zip(flist,range(len(flist))):
    with gzip.open(dataloc+str(f), 'r') as f_in, open(dataloc+flist[a][0:29], 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
for f in flist:
    os.remove(dataloc+str(f))

def writedate(first):
    with open('todaydate.txt', 'w') as f:
        f.write(str(first)+"\n")
writedate(first)
