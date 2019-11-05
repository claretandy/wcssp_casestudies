import os, sys
import datetime as dt
import matplotlib
####
# Use this for running on SPICE ...
hname = os.uname()[1]
if not hname.startswith('eld') and not hname.startswith('els') and not hname.startswith('vld'):
    matplotlib.use('Agg')
####
import iris
import iris.plot as iplt
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import datetime as dt
import glob
import re
import statistics
import subprocess
import cartopy.crs as ccrs
import matplotlib.colors as colors
sys.path.append('/net/home/h02/hadhy/Repository/hadhy_scripts/WCSSP/functions')
import std_functions as sf
import shutil
from pathlib import Path
import re
import pdb

print ('Hello World-------')
'''
Usage: python plot_timelagged.py 201810110000 201810130000 '96, -2, 108, 10' '120.5, 16.1, 122.1, 18.1'
This script takes as input a start date, an end date, an aggregation period and two bounding boxes.
The aggregation period refers to the period over which we want to make plots (default = 3 hours).
The difference between the start and end dates should be a multiple of the aggregation period (e.g. 24 hours would mean 8 plots are created, with aggtime of 3 hours)
The first bounding box is a plotting domain, the second is a sub-domain on which to calculate statistics.
Andy Hartley, October 2018
'''
def myround(x, base=3):
    return(int(base * np.floor(float(x)/base)))


def plotGPM(gpmlist, timeagg, odir):
    # Plot GPM data only
    return('hello')


def plotOneModel(gpmdict, modelcubes, model2plot, timeagg, plotdomain, odir):
    # Plot GPM against all lead times from one model

    try:
        m = [i for i, x in enumerate(modelcubes[model2plot]) if x][0]
        myu = modelcubes[model2plot][m].coord('time').units
        daterange = [x.strftime('%Y%m%dT%H%MZ') for x in myu.num2date(modelcubes[model2plot][m].coord('time').bounds[0])]
    except:
        pdb.set_trace()
        return


    if not os.path.isdir(odir):
        os.makedirs(odir)

    #pdb.set_trace()
    if len(modelcubes[model2plot]) < 10:
        diff = 10 - len(modelcubes[model2plot])
        # pdb.set_trace()
        modelcubes[model2plot] = np.repeat(None, diff).tolist() + modelcubes[model2plot]
        
    postage = {1 : gpmdict['gpm_prod_data'] if gpmdict['gpm_prod_data'] is not None else gpmdict['gpm_late_data'],
               2 : gpmdict['gpm_prod_qual'] if gpmdict['gpm_prod_qual'] is not None else gpmdict['gpm_late_qual'],
               3 : gpmdict['gpm_late_data'] if gpmdict['gpm_prod_data'] is not None else gpmdict['gpm_early_data'],
               4 : gpmdict['gpm_late_qual'] if gpmdict['gpm_prod_qual'] is not None else gpmdict['gpm_early_qual'],
               5 : modelcubes[model2plot][9],
               6 : modelcubes[model2plot][8],
               7 : modelcubes[model2plot][7],
               8 : modelcubes[model2plot][6],
               9 : modelcubes[model2plot][5],
               10 : modelcubes[model2plot][4],
               11 : modelcubes[model2plot][3],
               12 : modelcubes[model2plot][2],
               13 : modelcubes[model2plot][1],
               14 : modelcubes[model2plot][0],
               15 : None,
               16 : None
               }

    contour_levels = {'30mins': [0.0, 0.1, 0.25, 0.5, 1.0,  2.0,  4.0,  8.0, 16.0,  32.0, 1000.0],
                      '3hr'   : [0.0, 0.3, 0.75, 1.5, 3.0,  6.0, 12.0, 24.0, 48.0,  96.0, 1000.0],
                      '6hr'   : [0.0, 0.6, 1.5,  3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '12hr'  : [0.0, 0.6, 1.5,  3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '24hr'  : [0.0, 0.6, 1.5,  3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0]}

    my_rgb = ['#ffffff', '#87bbeb', '#6a9bde', '#2a6eb3', '#30ca28', '#e2d942', '#f49d1b', '#e2361d', '#f565f5',
              '#ffffff']

    # contour_levels = {'30mins': [1,2,4,8,16,32,64,128],
    #                   # '3hr'   : [1,2,4,8,16,32,64,128],
    #                   '3hr'   : [0,0.3,0.75,1.5,3.0,6.0,12.0,24.0,48.0,96.0],
    #                   '6hr'   : [1,2,4,8,16,32,64,128],
    #                   '12hr'  : [1,2,4,8,16,32,64,128,256],
    #                   '24hr'  : [1,2,4,8,16,32,64,128,256]} # TODO: Change thresholds especially for 12 & 24hr

    # my_rgb = ['#ffffff', '#87bbeb', '#5b98d6', '#276abc', '#1fc91a', '#fded39', '#f59800', '#eb2f1a', '#fe5cfc', '#ffffff'] # light blue '#6a9bde', mid blue
    bounds = contour_levels[str(timeagg)+'hr']
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(my_rgb))
    my_cmap = colors.ListedColormap(my_rgb)

    qbounds = np.arange(1,101)
    qnorm = colors.BoundaryNorm(boundaries=qbounds, ncolors=len(qbounds)-1)
    
    # Create a wider than normal figure to support our many plots
    #fig = plt.figure(figsize=(8, 12), dpi=100)
    fig = plt.figure(figsize=(12, 13), dpi=100)

    # Also manually adjust the spacings which are used when creating subplots
    plt.gcf().subplots_adjust(hspace=0.07, wspace=0.05, top=0.92, bottom=0.05, left=0.075, right=0.925)

    # iterate over all possible latitude longitude slices
    for i in range(1,17):
        #print(i)
        # plot the data in a 4 row x 4 col grid
        # pdb.set_trace()
        if postage[i] is not None:
            ax = plt.subplot(4, 4, i)
            # timebnds = postage[i].coord('time').bounds[0]
            # timediff = int(round(timebnds[1] - timebnds[0]))
            # plotthis = True if timediff == timeagg else False
            # if plotthis:
            if not (postage[i].coord('longitude').has_bounds() or postage[i].coord('latitude').has_bounds()):
                postage[i].coord('longitude').guess_bounds()
                postage[i].coord('latitude').guess_bounds()

            if 'Quality Flag' in postage[i].attributes['title']:
                qcm = iplt.pcolormesh(postage[i], vmin=0, vmax=100, cmap='cubehelix_r') #cmap='cubehelix_r') norm=qnorm,
            else:
                pcm = iplt.pcolormesh(postage[i], norm=norm, cmap=my_cmap) #cmap='cubehelix_r')

            # add coastlines
            ax = plt.gca()
            x0, y0, x1, y1 = plotdomain
            ax.set_extent([x0, x1, y0, y1])

            if i > 4:
                fclt = postage[i].coord('forecast_period').bounds[0]
                plt.title('T+'+str(int(fclt[0]))+' to '+'T+'+str(int(fclt[1])))
            else:
                plt.title(postage[i].attributes['title'])

            borderlines = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_0_boundary_lines_land',
                scale='50m',
                facecolor='none')

            ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
            ax.coastlines(resolution='50m', color='black')
            ax.gridlines(color="gray", alpha=0.2)
            #plt.gca().coastlines()

    # make an axes to put the shared colorbar in
    colorbar_axes = plt.gcf().add_axes([0.54, 0.1, 0.35, 0.025]) # left, bottom, width, height
    colorbar = plt.colorbar(pcm, colorbar_axes, orientation='horizontal', extend='neither')
    try:
        # colorbar.set_label('%s' % postage[i-2].units)
        colorbar.set_label('Precipitation amount (mm)')
    except:
        while postage[i-2] is None:
            i -= 1

    # Make another axis for the quality flag colour bar
    qcolorbar_axes = plt.gcf().add_axes([0.54, 0.2, 0.35, 0.025]) # left, bottom, width, height
    qcolorbar = plt.colorbar(qcm, qcolorbar_axes, orientation='horizontal')
    qcolorbar.set_label('Quality Flag')

    # limit the colorbar to 8 tick marks
    #import matplotlib.ticker
    #colorbar.locator = matplotlib.ticker.MaxNLocator(8)
    #colorbar.update_ticks()

    # get the time for the entire plot
    #time_coord = last_timestep.coord('time')
    #time = time_coord.units.num2date(time_coord.bounds[0, 0])

    # set a global title for the postage stamps with the date formated by
    # The following lines get around the problem of some missing data
    j = 0
    while modelcubes[model2plot][j] is None:
        j += 1
    newu = modelcubes[model2plot][j].coord('time').units
    key4p4 = [x for x in modelcubes.keys() if 'km4p4' in x][0] # This just gets the key for the 4.4km model (it changes depending on version)
    # pdb.set_trace()
    k = [i for i, x in enumerate(modelcubes[key4p4]) if x][0]
    daterng = [x.strftime('%Y%m%dT%H%MZ') for x in newu.num2date(modelcubes[key4p4][k+1].coord('time').bounds[0])]
    ofile = odir + 'plotOneModel_' + model2plot + '_' + daterng[0] + '-' + daterng[1] + '_timeagg' + str(
        timeagg) + 'hrs.png'
    # pdb.set_trace()
    plt.suptitle('Precipitation: GPM compared to %s for\n%s to %s' % (model2plot, daterng[0], daterng[1]), fontsize=18)

    fig.savefig(ofile, bbox_inches='tight')
    plt.close(fig)

    return ofile


def makeHTML_fromimages_indir(dt_start, dt_end, domain, pngfilelist, indir, eventname, timeagg):
    # makeHTML_fromimages_indir(dt_start, dt_end, pngfilelist, odir, 'timeagg' + str(timeagg) + 'hrs')
    # Make quick html page with all outputs

    outhtml = indir + '/precip_timelagged_'+timeagg+'.html'
    ifiles = [file for file in pngfilelist if file]

    try:
        imgdates = list(set([os.path.basename(file).split('_')[-2].split('-')[1] for file in ifiles if file]))
    except:
        print('There was a problem getting the imgdates from the list of files')
        return

    # Get the maximum number of side-by-side images (for the CSS column width)
    imgnum = 0
    for imgdate in sorted(imgdates):
        thesefiles = [ifile for ifile in ifiles if '-'+imgdate in ifile]
        imgnum = len(thesefiles) if len(thesefiles) > imgnum else imgnum

    if imgnum == 0:
        pdb.set_trace()

    htmlpage = open(outhtml, 'w')
    htmlpage.write('<!DOCTYPE html>\n')
    htmlpage.write('<html>\n')
    htmlpage.write('<head>\n')
    # htmlpage.write('  <link rel="stylesheet" href="'+outcss+'">\n')
    htmlpage.write('<style>\n')
    htmlpage.write('.column\n')
    htmlpage.write('{\n')
    htmlpage.write('float: left;\n')
    htmlpage.write('width: '+str(round(100./imgnum)-1)+' %;\n')
    htmlpage.write('padding: 0px;\n')
    htmlpage.write('}\n\n')

    htmlpage.write('.row::after\n')
    htmlpage.write('{\n')
    htmlpage.write('content: "";\n')
    htmlpage.write('clear: both;\n')
    htmlpage.write('display: table;\n')
    htmlpage.write('}\n')
    htmlpage.write('</style>\n')
    htmlpage.write('</head>\n')

    htmlpage.write('<body>\n')
    htmlpage.write('<h1>'+domain+'Case Study: '+eventname+' ('+timeagg.replace('timeagg','')+' aggregation)</h1>\n')
    htmlpage.write('<h2>For the period ' + dt_start.strftime('%a %d %B %Y (%H:%MUTC)') + ' to ' + dt_end.strftime('%a %d %B %Y (%H:%MUTC)') + '</h2>\n')

    imgnum = 0
    for imgdate in sorted(imgdates):
        # print(imgdate)
        thesefiles = [ifile for ifile in ifiles if '-'+imgdate in ifile]
        htmlpage.write('<div class="row">\n')
        htmlpage.write('<h3>'+imgdate+'</h3>\n')
        imgnum = len(thesefiles) if len(thesefiles) > imgnum else imgnum
        for this in thesefiles:
            htmlpage.write('<div class="column">\n')
            htmlpage.write('<figure>\n')
            # pdb.set_trace()
            htmlpage.write('<img src="'+this.replace(indir+'/', '')+'">\n') # style="width:100%"
            htmlpage.write('<figcaption>'+this.replace(indir+'/plots/', '')+'</figcaption>\n')
            htmlpage.write('</figure>\n')
            htmlpage.write('</div>\n')

        htmlpage.write('</div>\n')

    htmlpage.write('</body>\n')
    htmlpage.write('</html>\n')
    htmlpage.close()

    print('Output saved to: www-hc/~hadhy/CaseStudies/'+domain.lower()+'/'+eventname+'/'+os.path.basename(outhtml))

    return(outhtml)

def nice_names(name):

    nndict = {'seasia'  : 'SE Asia',
              'tafrica' : 'Tropical Africa'}

    try:
        return nndict[name]
    except:
        return name


def create_summary_html(indir):

    summarypage = Path(indir).as_posix() + '/index.html'
    htmlfiles = glob.glob(Path(indir).as_posix() + '/*/*/*.html')
    gpmfiles = glob.glob(os.path.dirname(summarypage) + '/*/*/gpm/gpm_30mins_current.html')
    htmlfiles.extend(gpmfiles)
    regions = [hf.replace(Path(indir).as_posix(), '').split('/')[1] for hf in htmlfiles]
    regions = sorted(list(set(regions)))
    eventlist = []

    # Create a summary html page on which to add reference to this page

    htmlpage = open(summarypage, 'w')
    htmlpage.write('<!DOCTYPE html>\n')
    htmlpage.write('<html>\n')
    htmlpage.write('<head>\n')
    htmlpage.write('<style>\n')
    htmlpage.write('table { font-family: arial, sans-serif; border-collapse: collapse; width: 100%; }\n')
    htmlpage.write('td, th { border: 1px solid #dddddd; text-align: left; padding: 8px; }\n')
    htmlpage.write('tr:nth-child(even) { background-color: #dddddd; }\n')
    htmlpage.write('</style>\n')
    htmlpage.write('</head>\n')
    htmlpage.write('<body>\n')

    htmlpage.write('<h1>List of Case Study web pages available</h1>\n')

    for reg in regions:
        htmlpage.write('<h2>'+nice_names(reg)+'</h2>\n')
        htmlpage.write('<a id="'+reg+'"></a>\n')

        reghtmlfiles = glob.glob(Path(summarypage).with_name(reg).as_posix() + '/**/*.html')
        gpmfiles = glob.glob(os.path.dirname(summarypage) + '/' + reg + '/*/gpm/gpm_30mins_current.html')
        reghtmlfiles.extend(gpmfiles)
        regbase = [x.replace(os.path.dirname(summarypage) + '/' + reg + '/', '') for x in reghtmlfiles]

        regcases = sorted(list(set([x.split('/')[0] for x in regbase])), reverse=True)

        htmlpage.write('<table>\n')
        htmlpage.write('<tr><th>Case Study Name</th><th>GPM</th><th>Models@3hrs</th><th>Models@6hrs</th><th>Models@12hrs</th><th>Models@24hrs</th></tr>\n')

        for rc in regcases:

            # get all the html files in this Region/Case combination
            allhtml = [rh for rh in reghtmlfiles if '/'+rc+'/' in rh]
            try:
                timeaggs = [os.path.basename(x).split('.')[0].split('timeagg')[1] for x in allhtml if not 'gpm' in x]
            except:
                pdb.set_trace()

            newline = '<tr><td>'+rc+'</td><td>GPM</td><td>3hrs</td><td>6hrs</td><td>12hrs</td><td>24hrs</td></tr>\n'
            # newline = '<p>' + rc + ': GPM 3hrs 6hrs 12hrs 24hrs </p>\n'

            for ta, html in zip(timeaggs, allhtml):
                insert = ' <a href="' + reg + '/' + rc + '/' + os.path.basename(html) + '">' + ta + '</a> '
                newline = newline.replace(ta, insert)

            # Get all the 30min GPM html file for this reg/case if it exists
            gpmhtmlfile = os.path.dirname(summarypage) + '/' + reg + '/' + rc + '/gpm/gpm_30mins_current.html'
            if os.path.isfile(gpmhtmlfile):
                insert = ' <a href="' + reg + '/' + rc + '/gpm/gpm_30mins_current.html' + '">GPM</a> '
                newline = newline.replace('GPM', insert)

            # Replace all cells with no data with a '-'
            newline = re.sub('<td>(GPM|[0-9]{1,}hrs)</td>', '<td>-</td>', newline)
            htmlpage.write(newline)

        htmlpage.write('</table>\n')

    htmlpage.write('</body>\n')
    htmlpage.write('</html>\n')
    htmlpage.close()



def plotAllData(gpmlist, modelcubes, models2plot, timeagg, odir):
    # Plot all the models and all the GPM data
    print('hello')
    
def main(dt_start, dt_end, timeagg, plotdomain, statdomain, searchlist=None, eventname=None, overwrite=False):
    # control the program

    print(timeagg)
    domain = sf.getDomain_bybox(plotdomain)
    datadir = '/data/users/hadhy/CaseStudies/'
    odir = datadir + domain.lower() + '/'
    html_odir = odir + eventname
    png_odir = html_odir + '/plots/'
    os.makedirs(png_odir, exist_ok=True)
    pngfilelist = []
    modellist = sf.getModels_bybox(plotdomain)['model_list']
    if searchlist:
        modellist = [ml for ml in modellist if ml in searchlist]

    # Loop through all timeagg-hour segments within the datetime range
    start = dt_start
    end = start + dt.timedelta(hours=timeagg)
    while end <= dt_end:
        
        print(start,' to ',end)
        # Load GPM Data
        
        try:
            gpm_prod_data, gpm_prod_qual = sf.getGPMCube(start, end, 'production', plotdomain)
        except:
            print('No GPM Production data available')

        try:
            gpm_late_data, gpm_late_qual = sf.getGPMCube(start, end, 'NRTlate', plotdomain)
        except:
            print('No GPM NRT Late data available')
            
        try:
            gpm_early_data, gpm_early_qual = sf.getGPMCube(start, end, 'NRTearly', plotdomain, aggregate=True)
        except:
            print('No GPM NRT Early data available')
            
        gpmdict = { "gpm_prod_data" : gpm_prod_data if 'gpm_prod_data' in locals() else None ,
                    "gpm_prod_qual" : gpm_prod_qual if 'gpm_prod_qual' in locals() else None ,
                    "gpm_late_data" : gpm_late_data if 'gpm_late_data' in locals() else None ,
                    "gpm_late_qual" : gpm_late_qual if 'gpm_late_qual' in locals() else None ,
                    "gpm_early_data" : gpm_early_data if 'gpm_early_data' in locals() else None ,
                    "gpm_early_qual" : gpm_early_qual if 'gpm_early_qual' in locals() else None }

        jobid = sf.getJobID_byDateTime(start, domain=domain, choice='newest')

        modelcubes = {}
        for mod in modellist:
            stash = sf.getPrecipStash(mod, type='short')
            lbproc = 128
            # args: loadModelData(start, end, stash, plotdomain, timeagg, model_id, jobid, odir, lbproc, overwrite=False
            print("Overwrite: ", overwrite)
            # start, end, stash, plotdomain, searchtxt = None, lbproc = 0, overwrite = False
            try:
                modelcubes[mod] = sf.loadModelData(start, end, stash, plotdomain, mod, lbproc, aggregate=True, overwrite=overwrite)
            except:
                continue
        
        # Do plotting
        # 1) Plot GPM
        #plotGPM(gpmdict, timeagg, odir)
        
        # 2) Plot GPM vs One Model
        for key in modelcubes.keys():
            pngfile = plotOneModel(gpmdict, modelcubes, key, timeagg, plotdomain, png_odir)
            pngfilelist.append(pngfile)

        # 3) GPM vs All Models
        #plotAllData(gpmdict, modelcubes, models2plot=[ga6, ga7, ra1_4p4, ra1_1p5_mal], timeagg, odir)
        
        # Set next start and end datetimes
        start = start + dt.timedelta(hours=timeagg)
        end = start + dt.timedelta(hours=timeagg)

    outhtml = makeHTML_fromimages_indir(dt_start, dt_end, domain, pngfilelist, html_odir, eventname, 'timeagg'+str(timeagg)+'hrs')

    create_summary_html(datadir)

if __name__ == '__main__':
    '''
    Usage:
    python plot_timelagged.py <start_dt> <end_dt> <aggregation_hrs> <plotdomain> <eventname> overwrite 
    '''

    try:
        dt_start = dt.datetime.strptime(sys.argv[1], "%Y%m%d%H%M") # Needs to be formatted %Y%m%d%H%M
        dt_end   = dt.datetime.strptime(sys.argv[2], "%Y%m%d%H%M") # Needs to be formatted %Y%m%d%H%M
    except IndexError:
        nrst3hour = myround(dt.datetime.now().hour, base=3)
        dt_end = dt.datetime.now().replace(hour=nrst3hour, minute=0, second=0, microsecond=0) - dt.timedelta(hours=6)
        dt_start = dt_end - dt.timedelta(hours=3)

    try:
        timeagg = int(sys.argv[3])
    except:
        timeagg = 3

    try:
        plotdomain = [float(x) for x in sys.argv[4].split(',')] # xmin, ymin, xmax, ymax
        #statdomain = [float(x) for x in sys.argv[5].split(', ')] # xmin, ymin, xmax, ymax
    except:
        # Assume a big SEAsia domain
        # latitude=(-10,20), longitude=(91, 120)
        # North  Vietnam: [102, 17, 108, 23]
        # Vietnam = [101, 4.5, 120, 24.5] # Vietnam
        plotdomain = [102, 17, 108, 23]

    try:
        searchlist = sys.argv[5]
    except:
        searchlist = 'ga6,km4p4'

    try:
        eventname = sys.argv[6]
    except:
        eventname = dt_start.strftime('%Y%m%d')

    try:
        overwrite = True if sys.argv[7] == 'overwrite' else False
    except:
        overwrite = False


    statdomain = plotdomain # [91, -10, 120, 20]
        
    main(dt_start, dt_end, timeagg, plotdomain, statdomain, searchlist=searchlist, eventname=eventname, overwrite=overwrite)
