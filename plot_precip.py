import os, sys
####
# Use this for running on SPICE ...
import matplotlib
matplotlib.use('Agg')
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
import iris.analysis as ia
import datetime as dt
import load_data
import location_config as config
import std_functions as sf
import run_html as html
import itertools
from downloadUM import get_local_flist
import pdb
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


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

def plotGPM(cube, region_name, location_name, bbox, bbox_name, overwrite=False, accum='12-hrs'):
    '''
    Plots GPM IMERG data for the given location and accumulation period
    :param cube:
    :param region_name:
    :param location_name:
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

    elif accum == '1-hr':
        # Aggregate by 3hr period
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube.aggregated_by(['day_of_year', 'hour'], iris.analysis.SUM) / 2.
        these_units = 'Accumulated rainfall (mm/hr)'

    elif accum == '30-mins':
        # Don't aggregate!
        # NB: Data is in mm/hr for each half hour timestep, so divide by 2
        cube_dom_acc = cube.copy()
        this_title = 'Rate (mm/hr) for 30-min Intervals'
        these_units = 'Accumulated rainfall (mm/hr)'
    else:
        print('Please specify a different accumulation time. Choose from: 30-mins, 1-hr, 3-hrs, 6-hrs, 12-hrs, 24-hrs')
        return

    cube_dom_acc.coord('latitude').guess_bounds()
    cube_dom_acc.coord('longitude').guess_bounds()

    contour_levels = {'30-mins': [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1000.0],
                      '1-hr': [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1000.0],
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
        timestamp = tpt.strftime('%Y%m%dT%H%MZ')

        # Make Output filename
        ofile = sf.make_outputplot_filename(region_name, location_name, tpt, 'GPM-IMERG', bbox_name,
                                            accum, 'Precipitation', 'Observed-Timeseries', 'T+0')

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
            lakelines = cfeature.NaturalEarthFeature(
                category='physical',
                name='lakes',
                scale='10m',
                edgecolor='black',
                alpha=0.5,
                facecolor='none')
            var_plt_ax.add_feature(lakelines)
            borderlines = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_0_boundary_lines_land',
                scale='50m',
                linewidth=1,
                linestyle=(0, (3, 1, 1, 1, 1, 1)),
                edgecolor='black',
                facecolor='none')
            var_plt_ax.add_feature(borderlines)
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
        return None

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

            lakelines = cfeature.NaturalEarthFeature(category='physical', name='lakes', scale='10m', edgecolor=cfeature.COLORS['water'], facecolor='none')
            ax.add_feature(lakelines)
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


def plotRegionalPrecipWind(analysis_data, gpm_data, region_bbox, region_bbox_name, settings, pstart, pend, time_tups, ofiles):
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

    try:
        cubex850_alltime = analysis_data['Uwind-levels'].extract(iris.Constraint(pressure=850.))
        cubey850_alltime = analysis_data['Vwind-levels'].extract(iris.Constraint(pressure=850.))
    except:
        return ofiles

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
    contour_levels = {'3-hrs': [0.3, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 1000.0],
                      '6-hrs': [0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '12-hrs': [0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '24-hrs': [0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '48-hrs': [0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '72-hrs': [0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '96-hrs': [0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0]}

    my_rgb = ['#87bbeb', '#6a9bde', '#2a6eb3', '#30ca28', '#e2d942', '#f49d1b', '#e2361d', '#f565f5', '#ffffff']

    diff = time_tups[0][1] - time_tups[0][0]
    timeagg = int(diff.seconds / (60 * 60))
    bounds = contour_levels[str(timeagg) + '-hrs']
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(my_rgb))
    my_cmap = colors.ListedColormap(my_rgb)

    # Temporarily build a data2plot dictionary so that we can get contour levels
    data2plot = {}
    myu = cubex850_alltime.coord('time').units
    for tpt in myu.num2date(cubex850_alltime.coord('time').points):
        data2plot[tpt] = {'analysis': {'Uwind': cubex850_alltime.extract(iris.Constraint(time=lambda t: t.point == tpt)),
                          'Vwind': cubey850_alltime.extract(iris.Constraint(time=lambda t: t.point == tpt))}}
    wspdcontour_levels = sf.get_contour_levels(data2plot, 'wind', extend='both', level_num=4)

    # Create a wider than normal figure to support our many plots (width, height)
    # fig = plt.figure(figsize=(8, 12), dpi=100)
    fig = plt.figure(figsize=(15, 12), dpi=100)

    # Also manually adjust the spacings which are used when creating subplots
    plt.gcf().subplots_adjust(hspace=0.07, wspace=0.05, top=0.91, bottom=0.075, left=0.075, right=0.8)

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

        plt.subplot(2, 2, i)
        pcm = iplt.pcolormesh(gpmdata_ss, norm=norm, cmap=my_cmap, zorder=2)

        # Set the plot extent
        ax = plt.gca()
        x0, y0, x1, y1 = region_bbox
        ax.set_extent([x0, x1, y0, y1], crs=crs_latlon)

        # Add a subplot title
        plt.title(tt[1].strftime('%Y%m%d %H:%M'))

        # Add Coastlines, Borders and Gridlines
        lakefill = cfeature.NaturalEarthFeature(
            category='physical',
            name='lakes',
            scale='50m',
            edgecolor='none',
            facecolor='lightgrey')
        ax.add_feature(lakefill, zorder=1)

        oceanfill = cfeature.NaturalEarthFeature(
            category='physical',
            name='ocean',
            scale='50m',
            edgecolor='none',
            facecolor='lightgrey')
        ax.add_feature(oceanfill, zorder=1)

        lakelines = cfeature.NaturalEarthFeature(
            category='physical',
            name='lakes',
            scale='50m',
            edgecolor='black',
            facecolor='none')
        ax.add_feature(lakelines, zorder=3)

        borderlines = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_boundary_lines_land',
            scale='50m',
            linewidth=1,
            linestyle=(0, (3, 1, 1, 1, 1, 1)),
            edgecolor='black',
            facecolor='none')
        ax.add_feature(borderlines, zorder=4)

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
        Y, X, U, V, spd, dir, lw = sf.compute_allwind(cubex850, cubey850, spd_levels=wspdcontour_levels)
        ax.streamplot(X, Y, U, V, density=1.5, color='k', linewidth=lw, zorder=5)

    # make an axes to put the shared colorbar in
    # colorbar_axes = plt.gcf().add_axes([0.175, 0.1, 0.65, 0.022])  # left, bottom, width, height
    colorbar_axes = plt.gcf().add_axes([0.85, 0.2, 0.022, 0.45])  # left, bottom, width, height
    colorbar = plt.colorbar(pcm, colorbar_axes, orientation='vertical', extend='max')
    colorbar.set_label('6-hr Precipitation Total (mm)')

    # make an axes to put the streamlines legend in
    strax = plt.gcf().add_axes([0.85, 0.7, 0.022, 0.15])  # left, bottom, width, height
    # colorbar = plt.colorbar(pcm, colorbar_axes, orientation='horizontal', extend='max')
    # colorbar.set_label('6-hr Precipitation Total (mm)')

    sf.makeStreamLegend(strax, wspdcontour_levels)

    # Use daterange in the title ...
    plt.suptitle('UM Analysis 850hPa winds and GPM IMERG Precipitation\n%s to %s' %
                 (pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M')), fontsize=18)

    region_name, location_name = [settings['region_name'], settings['location_name']]
    ofile = sf.make_outputplot_filename(region_name, location_name, pend, 'analysis', region_bbox_name,
                                        str(timeagg)+'-hrs', 'Precipitation', 'Regional-850winds', 'T+0')
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


def plot_postage(case_start, case_end, timeaggs, model_ids, region_name, location_name, bbox, bbox_name, settings, ofiles, max_plot_freq=12):
    '''
    Plots GPM IMERG (top row) vs model lead times for various time aggregations
    :param case_start: datetime for the start of the case study
    :param case_end: datetime for the end of the case study
    :param timeaggs: list of strings. Usually [3, 6, 12, 24, 48]
    :param model_ids: list of model_id strings
    :param region_name: string. Usually a larger region such as SEAsia or EastAfrica
    :param location_name: string. Usually the location of an event within the larger region
    :param bbox: list of coordinates [xmin, ymin, xmax, ymax]
    :param settings:
    :param ofiles: current list of output files
    :param max_plot_freq: int (hours). Sets the frequency at which the longer time aggregations are calculated (e.g. with max_plot_freq=12, a timeagg of 48hrs will be calculated every 00Z and 12Z)
    :return: List of output files created
    '''

    model_ids = [x for x in model_ids if x != 'analysis']
    plot_location = location_name

    for ta, mod in itertools.product(timeaggs, model_ids):

        time_segs = sf.get_time_segments(case_start, case_end, ta, max_plot_freq)

        for start, end in time_segs:
            print(ta, mod, start, end)

            # Load GPM IMERG data
            gpmdict = gpm_imerg_get_all(start, end, bbox, settings)

            # Load model data
            try:
                modelcubes = load_data.unified_model(start, end, settings, bbox=bbox, region_type='event', model_id=mod, var='precip', checkftp=False, timeclip=True, aggregate=True, totals=True)[mod]['precip']
            except:
                continue

            # Do the plotting for each
            # region_name, location_name, validtime, modelid, plot_location, timeagg, plottype, plotname, fclt, outtype='filesystem'
            plot_fname = sf.make_outputplot_filename(region_name, location_name, end, mod, bbox_name, str(ta)+'-hrs', 'Precipitation', 'Postage-Stamps', 'All-FCLT')

            try:
                pngfile = plotPostageOneModel(gpmdict, modelcubes, mod, ta, bbox, plot_fname)
                if pngfile:
                    ofiles.append(pngfile)
            except:
                pass

    return ofiles


def addTimeCats(cube):
    # Add day of year, hour of day, category of 12hr or 6hr
    iris.coord_categorisation.add_day_of_year(cube, 'time', name='day_of_year')
    sf.add_hour_of_day(cube, cube.coord('time'))
    sf.am_or_pm(cube, cube.coord('hour'))
    sf.accum_6hr(cube, cube.coord('hour'))
    sf.accum_3hr(cube, cube.coord('hour'))

    return cube


def plot_gpm(start, end, timeaggs, region_name, location_name, bbox, bbox_name, settings, ofiles):
    '''
    Plots GPM IMERG NRT_late and production for various time aggregations
    :param start: datetime
    :param end: datetime
    :param timeaggs: list of integers. Usually [3, 6, 12, 24]
    :param region_name: string. Formatted either region/datetime_location or monitoring/realtime_location
    :param location_name: string. Just the location name from the above
    :param bbox: list of floats or integers. Formatted [xmin, ymin, xmax, ymax]
    :param settings: dictionary. Created by the location_config.load_location_settings() function
    :param ofiles: current list of output files (this function adds to it)
    :return: List of output files created
    '''

    gpmdata = load_data.gpm_imerg(start, end, settings, latency='NRTlate', bbox=bbox, aggregate=False)
    gpmdata = addTimeCats(gpmdata)
    timeaggs = [tstr.replace('1-hrs','1-hr') for tstr in [str(t) + '-hrs' for t in timeaggs if type(t) == int]]

    for ta in timeaggs:

        ta_ofiles = plotGPM(gpmdata, region_name, location_name, bbox, bbox_name, overwrite=True, accum=ta)
        try:
            ofiles.extend(ta_ofiles)
        except:
            pass

    return ofiles


def plot_regional_plus_winds(start, end, model_ids, region_name, location_name, region_bbox, region_bbox_name, settings, ofiles):
    '''
    Plots GPM IMERG NRT_late (or production) with analysis winds vs model for various time slices
    2x2 plots of GPM precip + analysis winds (at 6 hour intervals)
    2x2 plots of model precip + model winds (at 6 hour intervals, by lead time, by model)
    :param start:
    :param end:
    :param model_ids:
    :param region_name:
    :param location_name:
    :param bbox:
    :param settings:
    :param ofiles: current list of output files
    :return: List of output files created
    '''

    # Remove 'analysis' from the model list
    model_ids = [m for m in model_ids if not m == 'analysis']

    # Get 24-hr time periods (at 6-hr intervals) for each plot
    plot_bnds = sf.get_time_segments(start, end, 24, max_plot_freq=6)

    for pstart, pend in plot_bnds:

        print(pstart, 'to', pend)

        # Get 6-hr time periods to cycle through
        time_tups = sf.get_time_segments(pstart, pend, 6)

        # Load the analysis wind data
        try:
            analysis_data = load_data.unified_model(pstart, pend, settings, bbox=(region_bbox + np.array([-5, -5, 5, 5])).tolist(), region_type='tropics', model_id='analysis', var=['Uwind-levels', 'Vwind-levels'], aggregate=False, timeclip=True)['analysis']
        except:
            continue

        # Load the GPM data
        try:
            gpm_data = load_data.gpm_imerg(pstart, pend, settings, bbox=region_bbox, aggregate=False)
        except:
            continue

        # Plot GPM & Analysis winds. 2x2 plots, 6-hr time slices
        ofiles = plotRegionalPrecipWind(analysis_data, gpm_data, region_bbox, region_bbox_name, settings, pstart, pend, time_tups, ofiles)

        # Plot T+24 for GPM&Analysis vs Model vs Difference (Rows: 4 time slices; Cols: obs, model, diff)


    return ofiles


def main(start=None, end=None, region_name=None, location_name=None, bbox=None):
    '''
    Loads data and runs all the precip plotting routines. The following variables are picked up from the settings dictionary
    :param start: datetime for the start of the case study
    :param end: datetime for the end of the case study
    :param region_name: String. Larger region E.g. 'SEAsia' or 'EastAfrica'
    :param location_name: String. Zoom area within the region. E.g. 'PeninsularMalaysia'
    :param bbox: List. Format [xmin, ymin, xmax, ymax]
    :return lots of plots
    '''

    settings = config.load_location_settings()
    if not start:
        start = settings['start']
    if not end:
        end = settings['end']
    if not region_name:
        region_name = settings['region_name']
    if not location_name:
        location_name = settings['location_name']
    if not bbox:
        bbox = settings['bbox']

    # Get the region plot bbox
    # NB: You can add to this by adding your own REGIONAL item to the dictionary in sf.getBBox_byRegionName
    reginfo = sf.getRegionBBox_byBBox(bbox)
    region_bbox = reginfo['region_bbox'] # sf.getBBox_byRegionName(sf.getModelDomain_bybox(bbox))

    # Make the start at 0000UTC of the first day and the end 0000UTC the last day
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end = (end + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # Set model ids to plot (by checking the available data on disk)
    full_file_list = get_local_flist(start, end, settings, region_type='event')
    model_ids = list(set([os.path.basename(fn).split('_')[1] for fn in full_file_list])) # ['analysis', 'ga7', 'km4p4', 'km1p5']

    # Time aggregation periods for all plots
    timeaggs = [1, 3, 6, 12, 24]  # 72, 96, 120

    # The names used to describe the plot area
    bbox_name = location_name
    region_bbox_name = reginfo['region_name']

    # Make an empty list for storing precip png plots
    ofiles = []

    # Run plotting functions
    ofiles = plot_postage(start, end, timeaggs, model_ids, region_name, location_name, bbox, bbox_name, settings, ofiles)
    ofiles = plot_gpm(start, end, timeaggs, region_name, location_name, bbox, bbox_name, settings, ofiles)
    ofiles = plot_gpm(start, end, timeaggs, region_name, location_name, region_bbox, region_bbox_name, settings, ofiles)
    ofiles = plot_regional_plus_winds(start, end, model_ids, region_name, location_name, region_bbox, region_bbox_name, settings, ofiles)

    html.create(ofiles)


if __name__ == '__main__':

    main()
