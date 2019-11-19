'''
Make quick near real time plots for a specfified time period.
Plots include:
    - Animated gif to go into powerpoint
    - 12-hr accumulations
    
Andy Hartley, August 2016
'''

import os, sys
####
# Use this for running on SPICE ...
import matplotlib
#hname = os.uname()[1]
#if not hname.startswith('eld') and not hname.startswith('els') and not hname.startswith('vld'):
#    matplotlib.use('Agg')
####
import iris
import iris.coord_categorisation
import iris.plot as iplt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import iris.quickplot as qplt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os.path
import shutil
import numpy.ma as ma
import iris.analysis as ia
from datetime import timedelta, date, datetime
import glob
from iris.coord_categorisation import add_categorised_coord
import re
#from PIL import Image
import itertools
import pdb

#iris.FUTURE.netcdf_promote = True
#iris.FUTURE.netcdf_no_unlimited = True


def most_common(lst):
    return max(set(lst), key=lst.count)


def getDataCube(strt, end, latency, quality=False):
    
    inpath = '/project/earthobs/PRECIPITATION/GPM/netcdf/imerg/'+latency+'/'
    
    if strt > end:
        raise ValueError('You provided a start_date that comes after the end_date.')
    
    file_search = inpath + '%Y/gpm_imerg_'+latency+'_*_%Y%m%d.nc' if not quality else inpath + '%Y/gpm_imerg_'+latency+'_*_%Y%m%d_quality.nc'
    numdays = (end - strt).days + 1
    file_list_wildcard = [(strt + timedelta(days=x)).strftime(file_search) for x in range(0, numdays)]
    file_list = []
    for fn in file_list_wildcard:
        file_part = glob.glob(fn.replace('.nc', '_part.nc'))
        file_full = glob.glob(fn)
        if len(file_full) == 1:
            file_list.extend(file_full)
        else:
            file_list.extend(file_part)
        
    newcubelist = []
    for file in file_list:
        if os.path.isfile(file):
            cube = iris.load_cube(file)
            # Make sure the order of the dims is correct
            mycoords = [c.name() for c in cube.coords()]
            if mycoords != ['time','latitude','longitude']:
                cube.transpose([mycoords.index('time'), mycoords.index('latitude'), mycoords.index('longitude')])
            newcubelist.append(cube)
        else:
            print('File (or part file) doesn\'t exist: ' + file)

    # Get Most common version
    version = most_common([os.path.basename(fn).split('_')[3] for fn in file_list])
    
    newcubelist = iris.cube.CubeList(newcubelist)
    try:
        cube = newcubelist.concatenate_cube()
    except:
        print('concatenate_cube failed')
        cube = newcubelist.concatenate()[0]
    cube.attributes['STASH'] = iris.fileformats.pp.STASH(1, 5, 216)
    cube.attributes['data_source'] = 'GPM'
    cube.attributes['product_name'] = 'imerg' if not quality else 'imerg quality flag'
    cube.attributes['latency'] = latency
    cube.attributes['version'] = version
    
    return(cube)


def domainClip(cube, domain):
    
    xmin, xmax, ymin, ymax = domain
    lonce = iris.coords.CoordExtent('longitude', xmin, xmax)
    latce = iris.coords.CoordExtent('latitude', ymin, ymax)
    #pdb.set_trace()
    cube_cropped = cube.intersection(lonce, latce)
    # cube_cropped.data = ma.masked_less(cube_cropped.data, 1)

    # Add day of year, hour of day, category of 12hr or 6hr
    iris.coord_categorisation.add_day_of_year(cube_cropped, 'time', name='day_of_year')
    add_hour_of_day(cube_cropped, cube_cropped.coord('time'))
    am_or_pm(cube_cropped, cube_cropped.coord('hour'))
    accum_6hr(cube_cropped, cube_cropped.coord('hour'))
    accum_3hr(cube_cropped, cube_cropped.coord('hour'))

    return(cube_cropped)


def add_hour_of_day(cube, coord, name='hour'):
    add_categorised_coord(cube, name, coord, lambda coord, x: coord.units.num2date(x).hour)

def am_or_pm(cube, coord, name='am_or_pm'):
    add_categorised_coord(cube, name, coord, lambda coord, x: 'am' if x < 12 else 'pm')

def accum_6hr(cube, coord, name='6hourly'):
    add_categorised_coord(cube, name, coord, lambda coord, x: 0 if x < 6 else 1 if x < 12 else 2 if x < 18 else 3)

def accum_3hr(cube, coord, name='3hourly'):
    add_categorised_coord(cube, name, coord, lambda coord, x: 0 if x < 3 else 1 if x < 6 else 2 if x < 9 else 3 if x < 12 else 4 if x < 15 else 5 if x < 18 else 6 if x < 21 else 7)


def plotGPM(cube_dom, outdir, domain, overwrite, accum='12hr'):

    print(accum + ' Accumulation')
    this_title = accum + ' Accumulation (mm)'
    ofilelist = []
    # print(np.nanmin(cube_dom.data))

    if accum == '24hr':
        # Aggregate by day_of_year.
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube_dom.aggregated_by('day_of_year', iris.analysis.SUM) / 2.
        these_units = 'Accumulated rainfall (mm/24hrs)'

    if accum == '12hr':
        # Aggregate by am or pm ...
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube_dom.aggregated_by(['day_of_year', 'am_or_pm'], iris.analysis.SUM) / 2.
        these_units = 'Accumulated rainfall (mm/12hrs)'

    if accum == '6hr':
        # Aggregate by 6hr period
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube_dom.aggregated_by(['day_of_year', '6hourly'], iris.analysis.SUM) / 2.
        these_units = 'Accumulated rainfall (mm/6hrs)'

    if accum == '3hr':
        # Aggregate by 3hr period
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube_dom.aggregated_by(['day_of_year', '3hourly'], iris.analysis.SUM) / 2.
        these_units = 'Accumulated rainfall (mm/3hrs)'

    if accum == '30mins':
        # Don't aggregate!
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube_dom.copy()
        this_title = 'Rate (mm/hr) for 30-min Intervals'
        these_units = 'Accumulated rainfall (mm/hr)'

    contour_levels = {'30mins': [0.0, 0.1, 0.25, 0.5, 1.0,  2.0,  4.0,  8.0, 16.0,  32.0, 1000.0],
                      '3hr'   : [0.0, 0.3, 0.75, 1.5, 3.0,  6.0, 12.0, 24.0, 48.0,  96.0, 1000.0],
                      '6hr'   : [0.0, 0.6, 1.5,  3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '12hr'  : [0.0, 0.6, 1.5,  3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '24hr'  : [0.0, 0.6, 1.5,  3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0]}

    my_rgb = ['#ffffff', '#87bbeb', '#6a9bde', '#2a6eb3', '#30ca28', '#e2d942', '#f49d1b', '#e2361d', '#f565f5', '#ffffff']
    
    # Plot the cube.
    for i in range(0,cube_dom_acc.shape[0]):
        print('Plotting ' + accum + ': ' + str(i+1) + ' / ' + str(cube_dom_acc.shape[0]))
        
        # Prepare time coords
        tcoord = cube_dom_acc[i].coord('time')
        tu = tcoord.units
        tpt = tu.num2date(tcoord.bounds[0][1]) + timedelta(seconds=1) # Get the upper bound and nudge it the hour

        # Define the correct output dir and filename
        thisyr = tpt.strftime('%Y')
        thismon = tpt.strftime('%m')
        thisday = tpt.strftime('%d')
        timestamp = tpt.strftime('%Y%m%dT%H%MZ')
        ofile   = outdir + thisyr + '/' + thismon + '/' + thisday + '/gpm-precip_'+accum+'_'+timestamp+'.png'

        print('Plotting ' + accum + ': ' + timestamp + ' (' + str(i) + ' / ' + str(cube_dom_acc.shape[0] - 1) + ')')

        if not os.path.isfile(ofile) or overwrite:
            
            this_localdir = os.path.dirname(ofile)
            if not os.path.isdir(this_localdir):
                os.makedirs(this_localdir)

            # Now do the plotting
            fig = plt.figure(figsize=getFigSize(domain), dpi=96)

            bounds = contour_levels[accum]
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(my_rgb))
            my_cmap = matplotlib.colors.ListedColormap(my_rgb)
            pcm = iplt.pcolormesh(cube_dom_acc[i], norm=norm, cmap=my_cmap)
            plt.title('GPM Precipitation ' + this_title + ' at\n' + tpt.strftime('%Y-%m-%d %H:%M'))
            plt.xlabel('longitude / degrees')
            plt.ylabel('latitude / degrees')
            var_plt_ax = plt.gca()

            var_plt_ax.set_extent(domain)
            # var_plt_ax.coastlines(resolution='50m')
            borderlines = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_0_boundary_lines_land',
                scale='50m',
                facecolor='none')
            var_plt_ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
            var_plt_ax.coastlines(resolution='50m', color='black')
            gl = var_plt_ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
            gl.xlabels_top = False
            gl.ylabels_left = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}

            vleft, vbottom, vwidth, vheight = var_plt_ax.get_position().bounds
            plt.gcf().subplots_adjust(top=vbottom+vheight, bottom=vbottom + 0.04,
                                      left=vleft, right=vleft + vwidth)
            cbar_axes = fig.add_axes([vleft, vbottom - 0.02, vwidth, 0.02])
            cbar = plt.colorbar(pcm, norm=norm, boundaries=bounds, cax=cbar_axes, orientation='horizontal', extend='both')
            cbar.set_label(these_units)
            cbar.ax.tick_params(length=0)

            fig.savefig(ofile, bbox_inches='tight')
            plt.close(fig)

        if os.path.isfile(ofile):
            # Add it to the list of files
            ofilelist = ofilelist + [ofile]
            # Make sure everyone can read it
            os.chmod(ofile, 0o777)
        
    return(ofilelist)


def move2web(filelist, local_dir):

    localfilelist = []
    
    for f in filelist:
        dt = datetime.strptime(os.path.basename(f).split('_')[2].split('.')[0], "%Y%m%dT%H%MZ")
        this_localdir = local_dir + dt.strftime("%Y") +'/'+ dt.strftime("%m") + '/' #+ dt.strftime("%d") + '/'

        if not os.path.isdir(this_localdir):
            os.makedirs(this_localdir)
            
        # Make symlink from remote_file to local_file
        if not os.path.isdir(this_localdir + dt.strftime("%d")):
            # os.symlink(src, dest)
            try:
                os.symlink(os.path.dirname(f), this_localdir + dt.strftime("%d"))
            except:
                print(os.path.dirname(f))
                print(this_localdir + dt.strftime("%d"))
                sys.exit('Problem with symlink on line 200.\nsrc: ' + os.path.dirname(f) + '\ndst: ' + this_localdir + dt.strftime("%d"))
            
        # Add local file to localfilelist
        localfilelist = localfilelist + [this_localdir + dt.strftime("%d") + '/' + os.path.basename(f)]

    return(localfilelist)


def getDomain(region_name):
    regname = region_name.lower()
    # xmin, xmax, ymin, ymax
    domains = {'seasia'     : [65, 160, -20, 30],
               'seasia_4k'  : [90, 154, -18, 30],
               'philippines': [100, 140, -5, 25],
               'malaysia'   : [99, 106, 0, 7], # Peninsula Malaysia
               'indonesia'  : [94, 154, -11, 6], # 
               'kuala_lumpur': [100.6, 102.6, 2.1, 4.1],
               'manila'     : [120, 122, 13.5, 15.5],
               'jakarta'    : [105.75, 107.75, -7.25, -5.25],
               'ghana'      : [-3.4, 1.4, 4.4, 11.4]
               }

    return(domains[regname])


def getFigSize(domain):
    xmin, xmax, ymin, ymax = domain
    ratio = (xmax - xmin) / (ymax - ymin)
    # print(ratio)
    ratiolut = [[0.4, (6, 12)],
                [0.8, (9, 12)],
                [1.2, (9, 9)],
                [1.6, (12, 9)],
                [1000., (12, 5)]]

    ofigsize = []
    for x in ratiolut:
        if ratio < x[0]:
            ofigsize = x[1]
            break

    return ofigsize


def getStartEnd(date_range, fmt):
    start_txt, end_txt = date_range.split('-')
    start = datetime.strptime(start_txt, '%Y%m%d').strftime(fmt)
    end   = datetime.strptime(end_txt, '%Y%m%d').strftime(fmt)
    return(start, end)

    
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


def writeHTML(ifiles, local_dir, template_file, out_html_file, dt_startdt, dt_enddt, timeperiod, region_name):
    
    url_base = "http://www-hc/~hadhy/"+region_name+"/gpm/"
    all_urls = [f.replace(local_dir, url_base) for f in ifiles]

    # Copy css
    
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

    proc_dt = datetime.utcnow()
    proctime = proc_dt.strftime("%H:%M %d/%m/%Y UTC")
    #start_date, end_date = getStartEnd(date_range, "%d/%m/%Y")

    # last image date
    lidt = datetime.strptime(os.path.basename(ifiles[-1]).split('_')[2].split('.')[0], "%Y%m%dT%H%MZ")
    
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

    # Make the symlink point to the most recently processed period
    olink = os.environ['HOME'] + '/public_html/' + region_name + '/gpm/gpm_' + timeperiod + '_current.html'
    if not os.path.islink(olink):
        os.symlink(out_html_file, olink)
    else:
        os.remove(olink)
        os.symlink(out_html_file, olink)

def mkOutDirs(dt_startdt, dt_enddt, outdir):
    # Make dirs for each year / month / day if they don't already exist
    for single_date in daterange(dt_startdt, dt_enddt + timedelta(days=1)):
        thisyear  = single_date.strftime("%Y")
        thismonth = single_date.strftime("%m")
        thisday   = single_date.strftime("%d")
        this_odir = outdir + thisyear + '/' + thismonth + '/' + thisday
        print(this_odir)
        if not os.path.isdir(this_odir):
            print('Creating ' + this_odir)
            os.makedirs(this_odir)


def main(dt_start, dt_end, region_name):

    # Set some things at the start ...
    overwrite = False
    template_file = '/home/h02/hadhy/Repository/hadhy_scripts/WCSSP/GPM/gpm_template.html'
    css_template = '/home/h02/hadhy/Repository/hadhy_scripts/WCSSP/GPM/style_gpm.css'
#    outdir = '/project/earthobs/PRECIPITATION/GPM/plots/'
    outdir = '/project/earthobs/PRECIPITATION/GPM/plots/' + region_name + '/'
    local_dir = os.environ['HOME'] + '/public_html/' + region_name + '/gpm/'
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    shutil.copyfile(css_template, local_dir + os.path.basename(css_template))
    
    # Change the start and end dates into datetime objects
    dt_startdt = datetime.strptime(dt_start, "%Y%m%d")
    dt_enddt = datetime.strptime(dt_end, "%Y%m%d") + timedelta(days=1) # Add an extra day so that we include the whole of the last day in the range
    dt_outhtml = datetime.strptime(dt_end, "%Y%m%d")

    # Make Output dirs
    mkOutDirs(dt_startdt, dt_enddt, outdir)

    domain = getDomain(region_name)
    cube = getDataCube(dt_startdt, dt_enddt, 'NRTearly')
    cube_dom = domainClip(cube, domain)

    accums = ['30mins', '3hr', '6hr', '12hr', '24hr'] #['12hr', '24hr']#
    
    for accum in accums:
        print(accum)
        filelist = plotGPM(cube_dom, outdir, domain, overwrite, accum)
        localfilelist = move2web(filelist, local_dir)
        out_html_file = outdir + dt_outhtml.strftime("%Y") +'/'+ dt_outhtml.strftime("%m") +'/'+ 'gpm_'+accum+'_'+dt_outhtml.strftime("%Y%m%dT%H%MZ")+'.html'
        writeHTML(localfilelist, local_dir, template_file, out_html_file, dt_startdt, dt_enddt, accum, region_name)


if __name__ == '__main__':
    dt_start = sys.argv[1]
    dt_end = sys.argv[2]
    region_name = sys.argv[3].lower() # seasia_4k
    
    #dt_start = '20170701'
    #dt_end   = '20170711'

    main(dt_start, dt_end, region_name)
    
