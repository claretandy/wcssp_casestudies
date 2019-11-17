import os, sys
import matplotlib
####
# Use this for running on SPICE in UKMO ...
hname = os.uname()[1]
if not hname.startswith('eld') and not hname.startswith('els'):
    matplotlib.use('Agg')
####
import location_config as config
import iris
import iris.coord_categorisation
import os.path
# from datetime import timedelta, date, datetime
import datetime as dt
import re
from PIL import Image
import shutil
import std_functions as sf
import nrt_plots_v3 as nrtplt
import plot_timelagged as pt

'''
Make quick near real time plots for a specfified time period.
Plots include:
    - Animated gif to go into powerpoint
    - 12-hr accumulations

Andy Hartley, August 2016
'''

def move2web(filelist, local_dir):

    localfilelist = []
    
    for f in filelist:
        this_dt = dt.datetime.strptime(os.path.basename(f).split('_')[2].split('.')[0], "%Y%m%dT%H%MZ")
        this_localdir = local_dir + this_dt.strftime("%Y") +'/'+ this_dt.strftime("%m") + '/' #+ dt.strftime("%d") + '/'

        if not os.path.isdir(this_localdir):
            os.makedirs(this_localdir)
            
        # Make symlink from remote_file to local_file
        if not os.path.isdir(this_localdir + this_dt.strftime("%d")):
            # os.symlink(src, dest)
            try:
                os.symlink(os.path.dirname(f), this_localdir + this_dt.strftime("%d"))
            except:
                print(os.path.dirname(f))
                print(this_localdir + this_dt.strftime("%d"))
                sys.exit('Problem with symlink on line 200.\nsrc: ' + os.path.dirname(f) + '\ndst: ' + this_localdir + this_dt.strftime("%d"))
            
        # Add local file to localfilelist
        localfilelist = localfilelist + [this_localdir + this_dt.strftime("%d") + '/' + os.path.basename(f)]

    return(localfilelist)


def getStartEnd(date_range, fmt):
    start_txt, end_txt = date_range.split('-')
    start = dt.datetime.strptime(start_txt, '%Y%m%d').strftime(fmt)
    end   = dt.datetime.strptime(end_txt, '%Y%m%d').strftime(fmt)
    return(start, end)

    
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)


def writeHTML(ifiles, local_dir, template_file, out_html_file, dt_startdt, dt_enddt, timeperiod, region_name):
    
    # url_base = "http://www-hc/~hadhy/seasia_4k/gpm_casestudies/"
    url_base = "http://www-hc/~hadhy/CaseStudies/"
    # all_urls = [f.replace(local_dir, url_base) for f in ifiles]
    all_urls = [url_base + f.split('CaseStudies/')[1] for f in ifiles]
    
    inline1='theImages[num] = new Image();\n'
    inline2='theImages[num].src = "url";\n'
    i_tot = str(len(ifiles) - 1)

    im = Image.open(ifiles[0])
    img_wd, img_ht = im.size # (width,height) tuple

    delaydict = {'30mins': 100,
                 '3hr'   : 1000,
                 '6hr'   : 1000,
                 '12hr'  : 1500,
                 '24hr'  : 2000}

    desc_txt_lu = {'30mins': 'Each timestep shows the mean precipitation rate (in mm/hour) for the 30 minutes preceeding the time shown.',
                   '3hr'   : 'Each timestep shows the accumulated precipitation for the 3 hours preceeding the time shown.',
                   '6hr'   : 'Each timestep shows the accumulated precipitation for the 6 hours preceeding the time shown.',
                   '12hr'  : 'Each timestep shows the accumulated precipitation for the 12 hours preceeding the time shown.',
                   '24hr'  : 'Each timestep shows the accumulated precipitation for the 24 hours preceeding the time shown.'
                   }

    proc_dt = dt.datetime.utcnow()
    proctime = proc_dt.strftime("%H:%M %d/%m/%Y UTC")
    #start_date, end_date = getStartEnd(date_range, "%d/%m/%Y")

    # last image date
    lidt = dt.datetime.strptime(os.path.basename(ifiles[-1]).split('_')[2].split('.')[0], "%Y%m%dT%H%MZ")
    
    with open(template_file, 'r') as input_file, open(out_html_file, 'w') as output_file:
        for line in input_file:
            if line.strip() == 'last_image=;':
                output_file.write('last_image='+i_tot+';\n')
            elif line.strip() == 'animation_height=;':
                output_file.write('animation_height='+str(img_ht)+';\n')
            elif line.strip() == 'animation_width=;':
                output_file.write('animation_width='+str(img_wd)+';\n')
            elif line.strip() == 'animation_startimg=;':
                output_file.write('animation_startimg="' + all_urls[0] +'";\n')
            elif re.search('insert-period', line.strip()):
                oline = line.replace('insert-period', date_range)
                output_file.write(oline)
            elif re.search('time-period', line.strip()):
                oline = line.replace('time-period', timeperiod)
                output_file.write(oline)
            elif re.search('mydelay', line.strip()):
                oline = line.replace('mydelay', str(delaydict[timeperiod]))
                output_file.write(oline)
            elif re.search('<li><a href="gpm_'+timeperiod+'_current.html">', line.strip()):
                oline = line.replace('a href','a class="current" href')
                output_file.write(oline)
            elif re.search('desc_txt', line.strip()):
                oline = line.replace('desc_txt', desc_txt_lu[timeperiod])
                output_file.write(oline)
            elif line.strip() == 'insert_imgdata_here':
                for iurl in all_urls:
                    # Replace the number of the image
                    i = all_urls.index(iurl)
                    oline1 = inline1.replace('num', str(i))
                    oline2 = inline2.replace('num', str(i))
                    # Replace the url
                    #this_url = 'http://www-hc/~hadhy/seasia_4k/gpm/' + date_range + '/archive/' + os.path.basename(ifile)
                    oline2 = oline2.replace('url', iurl)
                    # Write the lines out
                    output_file.write(oline1)
                    output_file.write(oline2)
            elif re.search('proctime', line.strip()):
                oline = line.replace('proctime', proctime)
                output_file.write(oline)
            elif re.search('start_date to end_date', line.strip()):
                #pdb.set_trace()
                oline = line.replace('start_date to end_date', dt_startdt.strftime("%Y-%m-%d %H:%MZ")+' to '+lidt.strftime("%Y-%m-%d %H%MZ"))
                output_file.write(oline)
            else:
                output_file.write(line)

    os.chmod(out_html_file, 0o777)
    
    # Copy the css file for the page style
    shutil.copyfile('/home/h02/hadhy/Repository/hadhy_scripts/WCSSP/functions/style_gpm.css', local_dir + 'style_gpm.css')

    # Make the symlink point to the most recently processed period
    olink = local_dir + 'gpm_' + timeperiod + '_current.html'
    if not os.path.islink(olink):
        os.symlink(out_html_file, olink)
    else:
        os.remove(olink)
        os.symlink(out_html_file, olink)


def mkOutDirs(dt_startdt, dt_enddt, outdir):
    # Make dirs for each year / month / day if they don't already exist
    for single_date in daterange(dt_startdt, dt_enddt + dt.timedelta(days=1)):
        thisyear  = single_date.strftime("%Y")
        thismonth = single_date.strftime("%m")
        thisday   = single_date.strftime("%d")
        this_odir = outdir + thisyear + '/' + thismonth + '/' + thisday
        print(this_odir)
        if not os.path.isdir(this_odir):
            print('Creating ' + this_odir)
            os.makedirs(this_odir)


def addTimeCats(cube):
    # Add day of year, hour of day, category of 12hr or 6hr
    iris.coord_categorisation.add_day_of_year(cube, 'time', name='day_of_year')
    nrtplt.add_hour_of_day(cube, cube.coord('time'))
    nrtplt.am_or_pm(cube, cube.coord('hour'))
    nrtplt.accum_6hr(cube, cube.coord('hour'))
    nrtplt.accum_3hr(cube, cube.coord('hour'))

    return cube


def main(dt_startdt, dt_enddt, plotdomain, region_name, eventname, organisation):

    # Set some things at the start ...
    settings = config.load_location_settings(organisation)
    rootdir = settings['plot_dir']

    # region_name = sf.getDomain_bybox(plotdomain).lower()
    template_file = 'gpm_template.html'
    outdir = rootdir + region_name + '/' + eventname + '/gpm/'
    local_dir = outdir
    #local_dir = os.environ['HOME'] + '/public_html/' + region_name + '/gpm_casestudies/'
    # if not os.path.isdir(local_dir):
    #     os.makedirs(local_dir)
    overwrite = True
    # TODO make plotdomains consistent across all code (perhaps dictionary item rather than a list)
    domain = [plotdomain[0], plotdomain[2], plotdomain[1], plotdomain[3]]
    
    # Change the start and end dates into datetime objects
    # dt_startdt = datetime.strptime(dt_start, "%Y%m%d")
    # dt_enddt = datetime.strptime(dt_end, "%Y%m%d") + dt.timedelta(days=1) # Add an extra day so that we include the whole of the last day in the range
    dt_outhtml = dt_enddt

    # Make Output dirs
    mkOutDirs(dt_startdt, dt_enddt, outdir)

    try:
        cube_dom = sf.getGPMCube(dt_startdt, dt_enddt, 'production', plotdomain, aggregate=False)
    except:
        cube_dom = sf.getGPMCube(dt_startdt, dt_enddt, 'NRTlate', plotdomain, aggregate=False)

    cube_dom = addTimeCats(cube_dom[0])
    accums = ['30mins', '3hr', '6hr', '12hr', '24hr'] #['12hr', '24hr']#
    
    for accum in accums:

        print(accum)

        filelist = nrtplt.plotGPM(cube_dom, outdir, domain, overwrite, accum)

        out_html_file = outdir + dt_outhtml.strftime("%Y") +'/'+ dt_outhtml.strftime("%m") +'/'+ 'gpm_'+accum+'_'+dt_outhtml.strftime("%Y%m%dT%H%MZ")+'.html'

        writeHTML(filelist, local_dir, template_file, out_html_file, dt_startdt, dt_enddt, accum, eventname)

    pt.create_summary_html(rootdir)

if __name__ == '__main__':

    try:
        dt_start = dt.datetime.strptime(sys.argv[1], "%Y%m%d%H%M") # Needs to be formatted %Y%m%d
        dt_end   = dt.datetime.strptime(sys.argv[2], "%Y%m%d%H%M") # Needs to be formatted %Y%m%d
    except IndexError:
        nrst3hour = sf.myround(dt.datetime.now().hour, base=3)
        dt_end = dt.datetime.now().replace(hour=nrst3hour, minute=0, second=0, microsecond=0) - dt.timedelta(hours=6)
        dt_start = dt_end - dt.timedelta(hours=3)

    try:
        plotdomain = [float(x) for x in sys.argv[3].split(',')] # xmin,ymin,xmax,ymax
    except:
        # Assume a big SEAsia domain
        plotdomain = [91, -10, 120, 25]

    try:
        eventname = sys.argv[4]
    except:
        eventname = 'noname/' + dt_start.strftime('%Y%m%d') + '_noname'

    region_name, eventname = eventname.split('/')

    try:
        organisation = sys.argv[5]
    except:
        organisation = 'Andy-MacBook'
    
    main(dt_start, dt_end, plotdomain, region_name, eventname, organisation)
