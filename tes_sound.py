import pandas as pd

# TODO Merge into plot_tephi.getObsData_BMKG()

upper_file = open("C:/Users/Freddy/Documents/GitHub/wcssp_casestudies/sample_upper.csv")
lines = upper_file.readlines()

lines = lines[2:]
upper_file.close()
id, lat, lon, date, dates, hr, press, t, td, dd, ff, name = [], [] ,[] ,[] ,[] ,[] ,[] ,[] ,[] ,[], [], []
for line in lines :
    line = line.replace('\n' ,'')
    line = line.replace(',', '.')
    line = line.split(';')
    id.append(float(line[0]))
    lat.append(float(line[1]))
    lon.append(float(line[2]))
    dates.append((str(line[3])[:8]))
    hr.append((str(line[3])[8:12]))
    press.append(float(line[5]))
    try:
        if float(line[6]) > -10000:
            ttt = float(line[6]) + 0
        else:
            ttt = "NA"
    except Exception:
        ttt = "NA"
    t.append(ttt)
    try:
        if float(line[7]) > -10000:
            tdt = float(line[7]) + 0
        else:
            tdt = "NA"
    except Exception:
        tdt = "NA"
    td.append(tdt)
    try:
        if float(line[8]) > -10000:
            ddd = float(line[8]) + 0
        else:
            ddd = "NA"
    except Exception:
        ddd = "NA"
    dd.append(ddd)
    try:
        if float(line[9]) > -10000:
            fff = float(line[9]) + 0
        else:
            fff = "NA"
    except Exception:
        fff = "NA"
    ff.append(fff)
    yymm = str((line[3])[:8])
    hhmm = str((line[3])[8:12])
    datem = yymm + "_" + hhmm
    named = "Station "+str(line[0])
    name.append(named)
    date.append(datem)



print (len(id))
print (len(lon))
print (len(lat))
print (len(t))
print (len(td))
print (len(dd))
print (len(ff))
print (len(date))

###Make DataFrame and save to CSV
df = pd.DataFrame(data={"Station Number": id, "Station Name": name, "Latitude": lat, "Longitude": lon, "Date": date, "Pressure":press, "temp": t, "TD": td, "DD": dd, "FF": ff})
df["Station Number"] = pd.to_numeric(id, errors='coerce')
df["Longitude"] = pd.to_numeric(lon, errors='coerce')
df["Latitude"] = pd.to_numeric(lat, errors='coerce')
df["temp"] = pd.to_numeric(t, errors='coerce')
df["Pressure"]= pd.to_numeric(press, errors='coerce')
df["TD"] = pd.to_numeric(td, errors='coerce')
df["DD"] = pd.to_numeric(dd, errors='coerce')
df["FF"] = pd.to_numeric(ff, errors='coerce')
##
df.to_csv("C:/Users/Freddy/Documents/GitHub/wcssp_casestudies/sample_upper_revised_bmkg.csv")
##
##print(df)
##
a = pd.read_csv("C:/Users/Freddy/Documents/GitHub/wcssp_casestudies/sample_upper_revised_bmkg.csv", sep=",", header=0, index_col=0, na_values=["NA",""]).dropna(how='any')
a.to_csv("C:/Users/Freddy/Documents/GitHub/wcssp_casestudies/sample_upper_revised_bmkg.csv", sep=",")

print ("Sounding Data Read          (OK)")
