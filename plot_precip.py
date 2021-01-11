import os, sys
####
# Use this for running on SPICE ...
import matplotlib
matplotlib.use('Agg')
# hname = os.uname()[1]
# if not hname.startswith('eld') and not hname.startswith('els') and not hname.startswith('vld'):
#    matplotlib.use('Agg')
####
import iris
import iris.coord_categorisation
import iris.plot as iplt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os.path
import shutil
import iris.analysis as ia
import datetime as dt
import re
import load_data
import location_config as config
import std_functions as sf
import run_html as html
import itertools
from downloadUM import get_local_flist
import pdb

def getFigSize(bbox):

    xmin, xmax, ymin, ymax = bbox
    ratio = (xmax - xmin) / (ymax - ymin)
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

def plotGPM(cube, event_name, event_location_name, bbox, overwrite=False, accum='12-hrs'):
    '''
    Plots GPM IMERG data for the given location and accumulation period
    :param cube:
    :param event_name:
    :param event_location_name:
    :param bbox:
    :param overwrite:
    :param accum:
    :return:
    '''

    print(accum + ' Accumulation')
    this_title = accum + ' Accumulation (mm)'
    ofilelist = []

    if accum == '24-hrs':
        # Aggregate by day_of_year.
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube.aggregated_by('day_of_year', iris.analysis.SUM) / 2.
        these_units = 'Accumulated rainfall (mm/24hrs)'

    elif accum == '12-hrs':
        # Aggregate by am or pm ...
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube.aggregated_by(['day_of_year', 'am_or_pm'], iris.analysis.SUM) / 2.
        these_units = 'Accumulated rainfall (mm/12hrs)'

    elif accum == '6-hrs':
        # Aggregate by 6hr period
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube.aggregated_by(['day_of_year', '6hourly'], iris.analysis.SUM) / 2.
        these_units = 'Accumulated rainfall (mm/6hrs)'

    elif accum == '3-hrs':
        # Aggregate by 3hr period
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube.aggregated_by(['day_of_year', '3hourly'], iris.analysis.SUM) / 2.
        these_units = 'Accumulated rainfall (mm/3hrs)'

    elif accum == '30-mins':
        # Don't aggregate!
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube.copy()
        this_title = 'Rate (mm/hr) for 30-min Intervals'
        these_units = 'Accumulated rainfall (mm/hr)'
    else:
        print('Please specify a different accumulation time. Choose from: 30-mins, 3-hrs, 6-hrs, 12-hrs, 24-hrs')
        return

    cube_dom_acc.coord('latitude').guess_bounds()
    cube_dom_acc.coord('longitude').guess_bounds()

    contour_levels = {'30-mins': [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1000.0],
                      '3-hrs': [0.0, 0.3, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 1000.0],
                      '6-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '12-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '24-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '48-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0]}

    my_rgb = ['#ffffff', '#87bbeb', '#6a9bde', '#2a6eb3', '#30ca28', '#e2d942', '#f49d1b', '#e2361d', '#f565f5',
              '#ffffff']

    # Plot the cube.
    for i in range(0, cube_dom_acc.shape[0]):

        # Prepare time coords
        tcoord = cube_dom_acc[i].coord('time')
        tu = tcoord.units
        tpt = tu.num2date(tcoord.bounds[0][1]) + dt.timedelta(seconds=1)  # Get the upper bound and nudge it the hour

        # Define the correct output dir and filename
        thisyr = tpt.strftime('%Y')
        thismon = tpt.strftime('%m')
        thisday = tpt.strftime('%d')
        timestamp = tpt.strftime('%Y%m%dT%H%MZ')
        # (event_name, this_dt_fmt, model_id, 'Tropics-' + lat0 + '-to-' + lat1,
        #  'Instantaneous', 'large-scale', 'walker-circulation', 'T+0')
        ofile = sf.make_outputplot_filename(event_name, timestamp, 'GPM-IMERG',
                 event_location_name, accum, 'Precipitation', 'Observed-Timeseries', 'T+0')

        print('Plotting ' + accum + ': ' + timestamp + ' (' + str(i+1) + ' / ' + str(cube_dom_acc.shape[0]) + ')')

        if not os.path.isfile(ofile) or overwrite:

            this_localdir = os.path.dirname(ofile)
            if not os.path.isdir(this_localdir):
                os.makedirs(this_localdir)

            # Now do the plotting
            fig = plt.figure(figsize=getFigSize(bbox), dpi=300)

            bounds = contour_levels[accum]
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(my_rgb))
            my_cmap = matplotlib.colors.ListedColormap(my_rgb)
            pcm = iplt.pcolormesh(cube_dom_acc[i], norm=norm, cmap=my_cmap)
            plt.title('GPM Precipitation ' + this_title + ' at\n' + tpt.strftime('%Y-%m-%d %H:%M'))
            plt.xlabel('longitude (degrees)')
            plt.ylabel('latitude (degrees)')
            var_plt_ax = plt.gca()

            var_plt_ax.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]])
            borderlines = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_0_boundary_lines_land',
                scale='50m',
                facecolor='none')
            var_plt_ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
            var_plt_ax.coastlines(resolution='50m', color='black')
            gl = var_plt_ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
            gl.top_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}

            vleft, vbottom, vwidth, vheight = var_plt_ax.get_position().bounds
            plt.gcf().subplots_adjust(top=vbottom + vheight, bottom=vbottom + 0.04,
                                      left=vleft, right=vleft + vwidth)
            cbar_axes = fig.add_axes([vleft, vbottom - 0.02, vwidth, 0.02])
            cbar = plt.colorbar(pcm, boundaries=bounds, cax=cbar_axes, orientation='horizontal', extend='both') # norm=norm,
            cbar.set_label(these_units)
            cbar.ax.tick_params(length=0)

            fig.savefig(ofile, bbox_inches='tight')
            plt.close(fig)


        if os.path.isfile(ofile):
            # Add it to the list of files
            ofilelist.append(ofile)
            # Make sure everyone can read it
            os.chmod(ofile, 0o777)

    return ofilelist


def plotPostageOneModel(gpmdict, modelcubes, model2plot, timeagg, plotdomain, ofile):
    # Plot GPM against all lead times from one model

    try:
        m = [i for i, x in enumerate(modelcubes) if x and i>0][0] # i>0 because sometimes the first record doesn't have the correct time span
        myu = modelcubes[m].coord('time').units
        daterange = [x.strftime('%Y%m%dT%H%MZ') for x in
                     myu.num2date(modelcubes[m].coord('time').bounds[0])]
    except:
        pdb.set_trace()
        return

    odir = os.path.dirname(ofile)
    if not os.path.isdir(odir):
        os.makedirs(odir)

    if len(modelcubes) < 10:
        diff = 10 - len(modelcubes)
        # pdb.set_trace()
        modelcubes = np.repeat(None, diff).tolist() + modelcubes

    postage = {1: gpmdict['gpm_prod_data'] if gpmdict['gpm_prod_data'] is not None else gpmdict['gpm_late_data'],
               2: gpmdict['gpm_prod_qual'] if gpmdict['gpm_prod_qual'] is not None else gpmdict['gpm_late_qual'],
               # TODO : Replace the next 2 lines with different observations either from satellite or radar
               3: gpmdict['gpm_late_data'] if gpmdict['gpm_prod_data'] is not None else gpmdict['gpm_early_data'],
               4: gpmdict['gpm_late_qual'] if gpmdict['gpm_prod_qual'] is not None else gpmdict['gpm_early_qual'],
               5: modelcubes[9],
               6: modelcubes[8],
               7: modelcubes[7],
               8: modelcubes[6],
               9: modelcubes[5],
               10: modelcubes[4],
               11: modelcubes[3],
               12: modelcubes[2],
               13: modelcubes[1],
               14: modelcubes[0],
               15: None,
               16: None
               }

    contour_levels = {'3-hrs': [0.0, 0.3, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 1000.0],
                      '6-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '12-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '24-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '48-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '72-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '96-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0]}

    my_rgb = ['#ffffff', '#87bbeb', '#6a9bde', '#2a6eb3', '#30ca28', '#e2d942', '#f49d1b', '#e2361d', '#f565f5',
              '#ffffff']

    bounds = contour_levels[str(timeagg) + '-hrs']
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(my_rgb))
    my_cmap = colors.ListedColormap(my_rgb)

    qbounds = np.arange(1, 101)
    # qnorm = colors.BoundaryNorm(boundaries=qbounds, ncolors=len(qbounds) - 1)

    # Create a wider than normal figure to support our many plots
    # fig = plt.figure(figsize=(8, 12), dpi=100)
    fig = plt.figure(figsize=(12, 13), dpi=100)

    # Also manually adjust the spacings which are used when creating subplots
    plt.gcf().subplots_adjust(hspace=0.07, wspace=0.05, top=0.92, bottom=0.05, left=0.075, right=0.925)

    # iterate over all possible latitude longitude slices
    for i in range(1, 17):
        # plot the data in a 4 row x 4 col grid
        if postage[i] is not None:
            ax = plt.subplot(4, 4, i)
            if not (postage[i].coord('longitude').has_bounds() or postage[i].coord('latitude').has_bounds()):
                postage[i].coord('longitude').guess_bounds()
                postage[i].coord('latitude').guess_bounds()
            if 'Quality Flag' in postage[i].attributes['title']:
                qcm = iplt.pcolormesh(postage[i], vmin=0, vmax=100, cmap='cubehelix_r')
            else:
                pcm = iplt.pcolormesh(postage[i], norm=norm, cmap=my_cmap)

            # add coastlines
            ax = plt.gca()
            x0, y0, x1, y1 = plotdomain
            ax.set_extent([x0, x1, y0, y1])

            if i > 4:
                fclt = postage[i].coord('forecast_period').bounds[0]
                plt.title('T+' + str(int(fclt[0])) + ' to ' + 'T+' + str(int(fclt[1])))
            else:
                plt.title(postage[i].attributes['title'])

            borderlines = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none')
            ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
            ax.coastlines(resolution='50m', color='black')
            ax.gridlines(color="gray", alpha=0.2)

    # make an axes to put the shared colorbar in
    colorbar_axes = plt.gcf().add_axes([0.54, 0.1, 0.35, 0.025])  # left, bottom, width, height
    colorbar = plt.colorbar(pcm, colorbar_axes, orientation='horizontal', extend='neither')
    try:
        # colorbar.set_label('%s' % postage[i-2].units)
        colorbar.set_label('Precipitation amount (mm)')
    except:
        while postage[i - 2] is None:
            i -= 1

    # Make another axis for the quality flag colour bar
    qcolorbar_axes = plt.gcf().add_axes([0.54, 0.2, 0.35, 0.025])  # left, bottom, width, height
    try:
        # NB: If there is no quality flag information available, we won't be able to plot the legend
        qcolorbar = plt.colorbar(qcm, qcolorbar_axes, orientation='horizontal')
        qcolorbar.set_label('Quality Flag')
    except:
        print('No quality flag data available')

    # Use daterange in the title ...
    plt.suptitle('Precipitation: GPM compared to %s for\n%s to %s' % (model2plot, daterange[0], daterange[1]), fontsize=18)

    fig.savefig(ofile, bbox_inches='tight')
    plt.close(fig)

    return ofile


def plotRegionalPrecipWind(analysis_data, gpm_data, region_bbox, settings, pstart, pend, time_tups, ofiles):
    '''

    :param analysis_data:
    :param gpm_data:
    :param region_bbox:
    :param settings:
    :param pstart:
    :param pend:
    :param time_tups:
    :param ofiles:
    :return:
    '''

    cubex850_alltime = analysis_data['Uwind-levels'].extract(iris.Constraint(pressure=850.))
    cubey850_alltime = analysis_data['Vwind-levels'].extract(iris.Constraint(pressure=850.))

    # First, make sure that we have data for the last of the 4 plots
    last_start, last_end = time_tups[-1]
    diff_hrs = (last_end - last_start).total_seconds()//3600
    # Get the length of all the input data
    gpmdata_ss = sf.periodConstraint(gpm_data, last_start, last_end)
    cubex850 = sf.periodConstraint(cubex850_alltime, last_start, last_end)
    cubey850 = sf.periodConstraint(cubey850_alltime, last_start, last_end)
    try:
        gpm_len_hrs = np.round(gpmdata_ss.coord('time').bounds[-1][1] - gpmdata_ss.coord('time').bounds[0][0])
        cubex_len_hrs = cubex850.coord('time').bounds[-1][1] - cubex850.coord('time').bounds[0][0]
        cubey_len_hrs = cubey850.coord('time').bounds[-1][1] - cubey850.coord('time').bounds[0][0]
    except:
        return ofiles
    if (diff_hrs != gpm_len_hrs) or (diff_hrs != cubex_len_hrs) or (diff_hrs != cubey_len_hrs):
        return ofiles

    # Create a figure
    contour_levels = {'3-hrs': [0.0, 0.3, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 1000.0],
                      '6-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '12-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '24-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '48-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '72-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '96-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0]}

    my_rgb = ['#ffffff', '#87bbeb', '#6a9bde', '#2a6eb3', '#30ca28', '#e2d942', '#f49d1b', '#e2361d', '#f565f5', '#ffffff']

    diff = time_tups[0][1] - time_tups[0][0]
    timeagg = int(diff.seconds / (60 * 60))
    bounds = contour_levels[str(timeagg) + '-hrs']
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(my_rgb))
    my_cmap = colors.ListedColormap(my_rgb)

    # Create a wider than normal figure to support our many plots (width, height)
    # fig = plt.figure(figsize=(8, 12), dpi=100)
    fig = plt.figure(figsize=(14.5, 12), dpi=100)

    # Also manually adjust the spacings which are used when creating subplots
    plt.gcf().subplots_adjust(hspace=0.07, wspace=0.05, top=0.92, bottom=0.15, left=0.075, right=0.925)

    # Set the map projection
    crs_latlon = ccrs.PlateCarree()

    # Loop through time_tups
    for tt in time_tups:

        i = time_tups.index(tt) + 1

        # Subset the GPM data
        try:
            gpmdata_ss = sf.periodConstraint(gpm_data, tt[0], tt[1])
            gpmdata_ss = gpmdata_ss.collapsed('time', iris.analysis.SUM)
            gpmdata_ss.data = gpmdata_ss.data / 2.
            gpmdata_ss.coord('latitude').guess_bounds()
            gpmdata_ss.coord('longitude').guess_bounds()
        except:
            print('Error getting GPM data')
            continue

        # Get the wind speed and line width
        cubex850 = sf.periodConstraint(cubex850_alltime, tt[0], tt[1])
        cubey850 = sf.periodConstraint(cubey850_alltime, tt[0], tt[1])

        if (not cubex850) or (not cubey850) :
            print('Error getting analysis data')
            continue

        Y = np.repeat(cubex850.coord('latitude').points[..., np.newaxis], cubex850.shape[1], axis=1)
        X = np.repeat(cubex850.coord('longitude').points[np.newaxis, ...], cubex850.shape[0], axis=0)
        U = cubex850.data
        V = cubey850.data
        speed = np.sqrt(U * U + V * V)
        lw = 5 * speed / speed.max()

        plt.subplot(2, 2, i)
        pcm = iplt.pcolormesh(gpmdata_ss, norm=norm, cmap=my_cmap)

        # Set the plot extent
        ax = plt.gca()
        x0, y0, x1, y1 = region_bbox
        ax.set_extent([x0, x1, y0, y1], crs=crs_latlon)

        # Add a subplot title
        plt.title(tt[1].strftime('%Y%m%d %H:%M'))

        # Add Coastlines, Borders and Gridlines
        borderlines = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none')
        ax.add_feature(borderlines, edgecolor='black', alpha=0.5)
        ax.coastlines(resolution='50m', color='black')
        gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)
        gl.top_labels = False
        if i == 1:
            gl.bottom_labels = False
            gl.right_labels = False
        elif i == 2:
            gl.bottom_labels = False
            gl.left_labels = False
        elif i == 3:
            gl.right_labels = False
        else:
            gl.left_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

        # Overlay wind field
        ax.streamplot(X, Y, U, V, density=1.5, color='k', linewidth=lw)

    # make an axes to put the shared colorbar in
    colorbar_axes = plt.gcf().add_axes([0.175, 0.1, 0.65, 0.022])  # left, bottom, width, height
    colorbar = plt.colorbar(pcm, colorbar_axes, orientation='horizontal', extend='max')
    colorbar.set_label('6-hr Precipitation Total (mm)')

    # Use daterange in the title ...
    plt.suptitle('UM Analysis 850hPa winds and GPM IMERG Precipitation\n%s to %s' % (pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M')), fontsize=18)

    ofile = sf.make_outputplot_filename(event_name, pend.strftime('%Y%m%dT%H%MZ'), 'analysis', event_location_name, str(timeagg)+'-hrs', 'Precipitation', 'Regional-850winds', 'T+0')
    fig.savefig(ofile, bbox_inches='tight')
    plt.close(fig)

    if os.path.isfile(ofile):
        ofiles.append(ofile)

    return ofiles


def gpm_imerg_get_all(start, end, bbox, settings):
    '''
    Runs the load_data.gpm_imerg function multiple times to create a dictionary of all available precip data and quality data
    :param start: datetime for start of the casestudy
    :param end: datetime for end of the casestudy
    :param bbox: list of [xmin, ymin, xmax, ymax] coords
    :return: dictionary for use in plottting
    '''

    try:
        gpm_prod_data = load_data.gpm_imerg(start, end, settings, latency='production', bbox=bbox, quality=False, aggregate=True)
        gpm_prod_qual = load_data.gpm_imerg(start, end, settings, latency='production', bbox=bbox, quality=True, aggregate=True)
    except:
        print('No GPM Production data available')

    try:
        gpm_late_data = load_data.gpm_imerg(start, end, settings, latency='NRTlate', bbox=bbox, quality=False, aggregate=True)
        gpm_late_qual = load_data.gpm_imerg(start, end, settings, latency='NRTlate', bbox=bbox, quality=True, aggregate=True)
    except:
        print('No GPM NRT Late data available')

    try:
        gpm_early_data = load_data.gpm_imerg(start, end, settings, latency='NRTearly', bbox=bbox, quality=False, aggregate=True)
        gpm_early_qual = load_data.gpm_imerg(start, end, settings, latency='NRTearly', bbox=bbox, quality=True, aggregate=True)
    except:
        print('No GPM NRT Early data available')

    gpmdict = {"gpm_prod_data": gpm_prod_data if 'gpm_prod_data' in locals() else None,
               "gpm_prod_qual": gpm_prod_qual if 'gpm_prod_qual' in locals() else None,
               "gpm_late_data": gpm_late_data if 'gpm_late_data' in locals() else None,
               "gpm_late_qual": gpm_late_qual if 'gpm_late_qual' in locals() else None,
               "gpm_early_data": gpm_early_data if 'gpm_early_data' in locals() else None,
               "gpm_early_qual": gpm_early_qual if 'gpm_early_qual' in locals() else None}

    return gpmdict


def get_time_segments(start, end, ta, max_plot_freq=12):
    '''
    Generates a list of tuples of start and end datetimes. The frequency (i.e. gap between tuples) is calculated either from ta (time aggregation period), or max_plot_freq, whichever is the smallest. For example, if ta=24 and max_plot_freq=12, then each tuple will be 24 hours from start to end, but that will be repeated every 12 hours.
    :param start: datetime for the start of the case study period
    :param end: datetime for the end of the case study period
    :param ta: integer. Time aggregation period
    :param max_plot_freq: integer. Maximum time difference between plots
    :return: list of tuples containing the start and end of each period
    '''
    step = min(ta, max_plot_freq)
    outlist = []

    # Make sure the start is anchored to a multiple of ta
    ## First, get a list of possible values
    tmpstart = start.replace(hour=0, minute=0, second=0, microsecond=0)
    xx = tmpstart
    poss_values = []
    while xx < tmpstart + dt.timedelta(days=1):
        poss_values.append(xx)
        xx = xx + dt.timedelta(hours=ta)
    ## Then, find the closest to the start
    stepstart = tmpstart
    while (stepstart + dt.timedelta(hours=ta)) <= start:
        stepstart += dt.timedelta(hours=ta)

    # Now, get the timeseries of tuples
    stepend = stepstart + dt.timedelta(hours=ta)
    while stepend <= end:
        outlist.append((stepstart, stepend))
        stepstart = stepstart + dt.timedelta(hours=step)
        stepend = stepstart + dt.timedelta(hours=ta)

    return outlist


def plot_postage(case_start, case_end, timeaggs, model_ids, event_name, event_location_name, bbox, settings, ofiles, max_plot_freq=12):
    '''
    Plots GPM IMERG (top row) vs model lead times for various time aggregations
    :param case_start: datetime for the start of the case study
    :param case_end: datetime for the end of the case study
    :param timeaggs: list of strings. Usually [3, 6, 12, 24, 48]
    :param model_ids: list of model_id strings
    :param event_name: string. Usually 'region/date_location'
    :param event_location_name: string. Usually the 'location' from the event_name string
    :param bbox: list of coordinates [xmin, ymin, xmax, ymax]
    :param settings:
    :param ofiles: current list of output files
    :param max_plot_freq: int (hours). Sets the frequency at which the longer time aggregations are calculated (e.g. with max_plot_freq=12, a timeagg of 48hrs will be calculated every 00Z and 12Z)
    :return: List of output files created
    '''

    model_ids = [x for x in model_ids if x != 'analysis']

    for ta, mod in itertools.product(timeaggs, model_ids):

        time_segs = get_time_segments(case_start, case_end, ta, max_plot_freq)

        for start, end in time_segs:
            print(ta, mod, start, end)

            # Load GPM IMERG data
            gpmdict = gpm_imerg_get_all(start, end, bbox, settings)

            # Load model data
            modelcubes = load_data.unified_model(start, end, event_name, settings, bbox=bbox, region_type='event', model_id=mod, var='precip', checkftp=False, timeclip=True, aggregate=True, totals=True)[mod]['precip']

            # Do the plotting for each
            plot_fname = sf.make_outputplot_filename(event_name, end.strftime('%Y%m%dT%H%MZ'), mod, event_location_name, str(ta)+'-hrs', 'Precipitation', 'Postage-Stamps', 'All-FCLT')
            pngfile = plotPostageOneModel(gpmdict, modelcubes, mod, ta, bbox, plot_fname)
            ofiles.append(pngfile)

    return ofiles


def addTimeCats(cube):
    # Add day of year, hour of day, category of 12hr or 6hr
    iris.coord_categorisation.add_day_of_year(cube, 'time', name='day_of_year')
    sf.add_hour_of_day(cube, cube.coord('time'))
    sf.am_or_pm(cube, cube.coord('hour'))
    sf.accum_6hr(cube, cube.coord('hour'))
    sf.accum_3hr(cube, cube.coord('hour'))

    return cube


def plot_gpm(start, end, timeaggs, event_name, event_location_name, bbox, settings, ofiles):
    '''
    Plots GPM IMERG NRT_late and production for various time aggregations
    :param start: datetime
    :param end: datetime
    :param timeaggs: list of integers. Usually [3, 6, 12, 24]
    :param event_name: string. Formatted either region/datetime_location or monitoring/realtime_location
    :param event_location_name: string. Just the location name from the above
    :param bbox: list of floats or integers. Formatted [xmin, ymin, xmax, ymax]
    :param settings: dictionary. Created by the location_config.load_location_settings() function
    :param ofiles: current list of output files (this function adds to it)
    :return: List of output files created
    '''

    gpmdata = load_data.gpm_imerg(start, end, settings, latency='NRTlate', bbox=bbox, aggregate=False)
    gpmdata = addTimeCats(gpmdata)
    timeaggs = [str(t) + '-hrs' for t in timeaggs if type(t) == int]

    # for ta, mod in itertools.product(timeaggs, model_ids):
    #     for start, end in get_time_segments(case_start, case_end, ta, max_plot_freq):
    #         print(ta, mod, start, end)

    for ta in timeaggs:

        ta_ofiles = plotGPM(gpmdata, event_name, event_location_name, bbox, overwrite=False, accum=ta)
        try:
            ofiles.extend(ta_ofiles)
        except:
            pass

    return ofiles


def plot_regional_plus_winds(start, end, model_ids, event_name, event_location_name, bbox, settings, ofiles):
    '''
    Plots GPM IMERG NRT_late (or production) with analysis winds vs model for various time slices
    2x2 plots of GPM precip + analysis winds (at 6 hour intervals)
    2x2 plots of model precip + model winds (at 6 hour intervals, by lead time, by model)
    :param start:
    :param end:
    :param model_ids:
    :param event_name:
    :param bbox:
    :param settings:
    :param ofiles: current list of output files
    :return: List of output files created
    '''

    # Get the region plot bbox
    # NB: You can add to this by adding your own REGIONAL item to the dictionary in sf.getBBox_byRegionName
    region_bbox = sf.getBBox_byRegionName(sf.getDomain_bybox(bbox))

    # Remove 'analysis' from the model list
    model_ids = [m for m in model_ids if not m == 'analysis']

    # Get 24-hr time periods (at 6-hr intervals) for each plot
    plot_bnds = get_time_segments(start, end, 24, max_plot_freq=6)

    for pstart, pend in plot_bnds:

        print(pstart, 'to', pend)

        # Get 6-hr time periods to cycle through
        time_tups = get_time_segments(pstart, pend, 6)

        # Load the analysis wind data
        try:
            analysis_data = load_data.unified_model(pstart, pend, event_name, settings, bbox=(region_bbox + np.array([-5, -5, 5, 5])).tolist(), region_type='event', model_id='analysis', var=['Uwind-levels', 'Vwind-levels'], aggregate=False, timeclip=True)['analysis']
        except:
            continue

        # Load the GPM data
        try:
            gpm_data = load_data.gpm_imerg(pstart, pend, settings, bbox=region_bbox, aggregate=False)
        except:
            continue

        # Plot GPM & Analysis winds. 2x2 plots, 6-hr time slices
        ofiles = plotRegionalPrecipWind(analysis_data, gpm_data, region_bbox, settings, pstart, pend, time_tups, ofiles)

        # Plot T+24 for GPM&Analysis vs Model vs Difference (Rows: 4 time slices; Cols: obs, model, diff)


    return ofiles


def main(start, end, event_name, event_location_name, bbox, organisation):
    '''
    Loads data and runs all the precip plotting routines
    :param start: datetime for the start of the case study
    :param end: datetime for the end of the case study
    :param event_name: String. Format <region>/<date>_<event_location_or_name> E.g. 'PeninsulaMalaysia/20200520_Johor'
    :param event_location_name: String. Location name for plotting
    :param bbox: List. Format [xmin, ymin, xmax, ymax]
    :param organisation: string.
    :return: lots of plots
    '''

    # Set some location-specific defaults
    settings = config.load_location_settings(organisation)

    # Get the region plot bbox
    # NB: You can add to this by adding your own REGIONAL item to the dictionary in sf.getBBox_byRegionName
    region_name = sf.getDomain_bybox(bbox)
    region_bbox = sf.getBBox_byRegionName(region_name)

    # Make the start at 0000UTC of the first day and the end 0000UTC the last day
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end = (end + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # Set model ids to plot (by checking the available data on disk)
    full_file_list = get_local_flist(start, end, event_name, settings, region_type='event')
    model_ids = list(set([os.path.basename(fn).split('_')[-3] for fn in full_file_list])) # ['analysis', 'ga7', 'km4p4', 'km1p5']

    # Time aggregation periods for all plots
    timeaggs = [3, 6, 12, 24]  # 72, 96, 120

    # Make an empty list for storing precip png plots
    ofiles = []

    # Run plotting functions
    ofiles = plot_postage(start, end, timeaggs, model_ids, event_name, event_location_name, bbox, settings, ofiles)
    ofiles = plot_gpm(start, end, timeaggs, event_name, event_location_name, bbox, settings, ofiles)
    ofiles = plot_gpm(start, end, timeaggs, event_name, region_name, region_bbox, settings, ofiles)
    ofiles = plot_regional_plus_winds(start, end, model_ids, event_name, event_location_name, bbox, settings, ofiles)

    html.create(ofiles)


if __name__ == '__main__':

    try:
        start = dt.datetime.strptime(sys.argv[1], '%Y%m%d%H%M')
    except:
        # For realtime
        start = dt.datetime.utcnow() - dt.timedelta(days=10)

    try:
        end = dt.datetime.strptime(sys.argv[2], '%Y%m%d%H%M')
    except:
        # For realtime
        end = dt.datetime.utcnow()

    try:
        event_name = sys.argv[3]
    except:
        # For realtime
        event_name = 'monitoring/realtime_Peninsular-Malaysia'

    try:
        event_location_name = sys.argv[4]
    except:
        # For realtime
        event_location_name = 'Peninsular-Malaysia'

    try:
        domain_str = sys.argv[5]
        bbox = [float(x) for x in domain_str.split(',')]
    except:
        # For testing
        bbox = [100, 0, 110, 10]

    try:
        organisation = sys.argv[6]
    except:
        organisation = 'UKMO'

    main(start, end, event_name, event_location_name, bbox, organisation)
