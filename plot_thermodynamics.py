import os
import matplotlib
matplotlib.use('Agg')
import iris
import iris.plot as iplt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import datetime as dt
import std_functions as sf
import location_config as config
from downloadUM import get_local_flist
# import plot_precip
import load_data
import run_html as html
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import numpy.ma as ma
import pdb
# import metpy

def uv_to_speed_dir(u, v):

    Y = np.repeat(u.coord('latitude').points[..., np.newaxis], u.shape[1], axis=1)
    X = np.repeat(u.coord('longitude').points[np.newaxis, ...], u.shape[0], axis=0)
    U = u.data
    V = v.data
    speed = np.sqrt(U**2 + V**2)
    lw = 5 * speed / speed.max() # Line width

    speed_cube = u.copy(speed)

    return Y, X, U, V, lw, speed_cube


def plotOLRPrecipLgtng(pstart, pend, data2plot, bbox, ofile, settings):
    '''
    Do the plotting of the data2plot
    :param pstart:
    :param pend:
    :param data2plot:
    :param bbox:
    :param ofile:
    :param settings:
    :return: plot filename
    '''

    print('Plotting data from: '+pstart.strftime('%Y%m%d %H:%M')+' to '+pend.strftime('%Y%m%d %H:%M'))
    # Read data2plot to setup plot dimensions
    ncols = 3
    row_dict = {}
    cnt = 1
    for k in np.arange(len(data2plot.keys())):
        row_dict[k] = np.arange(cnt, cnt + ncols)
        cnt += ncols

    nrows = len(row_dict.keys())
    obs_pos = np.arange(1, 20, 3) # [1, 4, 7, 10]
    reg_pos = np.arange(2, 21, 3) # [2, 5, 8, 11]
    glo_pos = np.arange(3, 22, 3) # [3, 6, 9, 12]

    contour_levels = {'3-hrs': [0.0, 0.3, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 1000.0],
                      '6-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '12-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '24-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '48-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '72-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '96-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0]}

    my_rgb = ['#ffffff', '#87bbeb', '#6a9bde', '#2a6eb3', '#30ca28', '#e2d942', '#f49d1b', '#e2361d', '#f565f5',
              '#ffffff']

    bounds = contour_levels['3-hrs']
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(my_rgb))
    my_cmap = colors.ListedColormap(my_rgb)

    fig = plt.figure(figsize=(16, 16), dpi=100)

    # Also manually adjust the spacings which are used when creating subplots
    plt.gcf().subplots_adjust(hspace=0.1, wspace=0.05, top=0.92, bottom=0.05, left=0.22, right=0.85)

    # iterate over all row datetimes
    for rowdt, data in data2plot.items():

        rowi = list(data2plot.keys()).index(rowdt)

        for i in row_dict[rowi]:
            # pdb.set_trace()
            ax = fig.add_subplot(nrows, ncols, i, projection=ccrs.PlateCarree())
            # ax = plt.gca()
            x0, y0, x1, y1 = bbox
            ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

            if i in obs_pos:
                ocm = iplt.pcolormesh(data2plot[rowdt]['obs-olr'], cmap='Greys')
                pcm = iplt.pcolormesh(data2plot[rowdt]['obs-gpm'], norm=norm, cmap=my_cmap)
                #TODO Add lightning obs points in here too
                plt.title('Observations', fontsize=14)

            if i in reg_pos:
                fclt = data2plot[rowdt]['um-regional-precip'].coord('forecast_period').bounds[0]
                iplt.pcolormesh(data2plot[rowdt]['um-regional-olr'], cmap='Greys')
                iplt.pcolormesh(data2plot[rowdt]['um-regional-precip'], norm=norm, cmap=my_cmap)
                levs = np.arange(0, data2plot[rowdt]['um-regional-lightning'].data.max()+3, 10)
                iplt.contour(data2plot[rowdt]['um-regional-lightning'], levels=levs)
                plt.title('UM-TA4: T+' + str(int(fclt[0])) + ' to ' + 'T+' + str(int(fclt[1])), fontsize=14)

            if i in glo_pos:
                fclt = data2plot[rowdt]['um-global-precip'].coord('forecast_period').bounds[0]
                iplt.pcolormesh(data2plot[rowdt]['um-global-olr'], cmap='Greys')
                iplt.pcolormesh(data2plot[rowdt]['um-global-precip'], norm=norm, cmap=my_cmap)
                plt.title('UM-Global: T+' + str(int(fclt[0])) + ' to ' + 'T+' + str(int(fclt[1])), fontsize=14)

            # Add Coastlines, Borders and Gridlines
            lakelines = cfeature.NaturalEarthFeature(
                category='physical',
                name='lakes',
                scale='10m',
                edgecolor='black',
                alpha=0.5,
                facecolor='none')
            ax.add_feature(lakelines)
            borderlines = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_0_boundary_lines_land',
                scale='50m',
                linewidth=1,
                linestyle=(0, (3, 1, 1, 1, 1, 1)),
                edgecolor='black',
                facecolor='none')
            ax.add_feature(borderlines)
            ax.coastlines(resolution='50m', color='black')
            gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)

            if rowi < (nrows - 1):
                gl.bottom_labels = False
                gl.top_labels = False
            else:
                gl.top_labels = False

            if i == row_dict[rowi][0]:
                gl.right_labels = False
                ax.text(-0.13, 0.5, rowdt.strftime('%Y%m%d %H:%M'), fontsize=14, va='center', ha='right', rotation='vertical', transform=ax.transAxes)
            elif i == row_dict[rowi][ncols-1]:
                gl.left_labels = False
            else:
                gl.right_labels = False
                gl.left_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}

    # Make an axes to put the shared colorbar in
    try:
        colorbar_axes = plt.gcf().add_axes([0.88, 0.5, 0.025, 0.35])  # left, bottom, width, height
        colorbar = plt.colorbar(pcm, colorbar_axes, orientation='vertical', extend='max') # va='center',
        colorbar.set_label('Precipitation accumulation (mm/3-hr)')
    except:
        pass

    # Make another axes for the OLR colour bar
    try:
        ocolorbar_axes = plt.gcf().add_axes([0.88, 0.1, 0.025, 0.35])  # left, bottom, width, height
        ocolorbar = plt.colorbar(ocm, ocolorbar_axes, orientation='vertical')
        ocolorbar.set_label('Outgoing Longwave Radiation (Wm-2)')
    except:
        pass

    # Use daterange in the title ...
    plt.suptitle('OLR, Precipitation and Lightning: Observations, UM-TA4 and UM-Global\n%s to %s (UTC)' % (pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M')), fontsize=18)

    fig.savefig(ofile, bbox_inches='tight')
    plt.close(fig)

    return ofile


def plot_olr_pr_lgt(start, end, bbox, ofiles, settings):
    '''
    Plot of 4 Rows x 3 Columns, where the rows are 4 datetimes (3-hourly) FROM THE SAME MODEL RUN,
    and the columns are Obs, regional model, global model. This plot overlays OLR, precip and lightning where available
    :param start:
    :param end:
    :param bbox:
    :param ofiles:
    :param settings:
    :return: list of ofiles that includes the created plot
    '''

    fchrs = sf.get_fc_InitHours(settings['jobid'])
    if not start.hour in fchrs or not end.hour in fchrs:
        start = start.replace(hour=fchrs[0])
        end = end.replace(hour=fchrs[0])

    # Most jobs run 12-hourly, either on 0 and 12Z or 6 and 18Z
    plot_freq = 12
    plot_time_range = sf.get_time_segments(start, end, 12, plot_freq, start_hour_zero=False)
    fclts = [(12,24), (24,36), (36,48)] # [(6, 18), (12,24), (18, 30), (24,36), (30,42)]

    for (pstart, pend), fclt in itertools.product(plot_time_range, fclts):

        print(pstart, pend, fclt)

        # Get 3-hr time periods to cycle through
        row_time_ranges = sf.get_time_segments(pstart, pend, 3)

        # Make an output dictionary for sending to the plotting routine
        data2plot = {}

        for rstart, rend in row_time_ranges:

            data2plot[rend] = {}

            # Load the GPM data
            # Aggregated for the period of the row (normally 3 hours)
            try:
                print('Loading GPM')
                data2plot[rend]['obs-gpm'] = load_data.gpm_imerg(rstart, rend, settings, bbox=bbox, aggregate=True)
            except:
                continue

            # Load OLR Observations
            # Instantaneous obs at the end of the period of the row
            # Initially from autosat, but could try to get satPy working too
            # TODO Add the satellite OLR data to the FTP upload
            try:
                print('Loading OLR obs')
                data2plot[rend]['obs-olr'] = load_data.satellite_olr(rstart, rend, settings, bbox=bbox, aggregate=False, timeseries=False)
            except:
                print("Problem loading satellite OLR data")
                continue

            # TODO Load the lightning obs

            # Load the UM Regional Model data
            regmod = 'africa-prods'
            try:
                print('Loading africa-prods')
                data2plot[rend]['um-regional-olr'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[regmod], var=['olr-toa'],
                                                  fclt_clip=fclt, aggregate=False, timeclip=True)[regmod]['olr-toa'][0][-1,...]
                data2plot[rend]['um-regional-precip'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[regmod], var=['precip'],
                                                  fclt_clip=fclt, aggregate=True, totals=True, timeclip=True)[regmod]['precip']
                data2plot[rend]['um-regional-lightning'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[regmod], var=['num-of-lightning-flashes'],
                                                  fclt_clip=fclt, aggregate=True, totals=True, timeclip=True)[regmod]['num-of-lightning-flashes']
            except:
                continue

            # Load the UM Global Model data
            glomod = 'global-prods'
            try:
                print("Loading global-prods")
                data2plot[rend]['um-global-olr'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[glomod], var=['olr-toa'],
                                                  fclt_clip=fclt, aggregate=False, timeclip=True)[glomod]['olr-toa'][0][-1,...]
                data2plot[rend]['um-global-precip'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[glomod], var=['precip'],
                                                  fclt_clip=fclt, aggregate=True, totals=True, timeclip=True)[glomod]['precip']
                data2plot[rend]['um-global-lightning'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[glomod], var=['num-of-lightning-flashes'],
                                                  fclt_clip=fclt, aggregate=True, totals=True, timeclip=True)[glomod]['num-of-lightning-flashes']
            except:
                continue

            print('Sorting out data2plot ...')
            for k in data2plot[rend].keys():
                if data2plot[rend][k] == []:
                    data2plot[rend][k] = None
                if isinstance(data2plot[rend][k], list) or isinstance(data2plot[rend][k], iris.cube.CubeList):
                    data2plot[rend][k] = data2plot[rend][k][0]
                if 'precip' in k:
                    data2plot[rend][k].data = ma.masked_less(data2plot[rend][k].data, 0.6)

        # Do the plotting for each
        # region_name, location_name, validtime, modelid, plot_location, timeagg, plottype, plotname, fclt, outtype='filesystem'
        plot_fname = sf.make_outputplot_filename(settings['region_name'], settings['location_name'], pend, 'Obs+Models', settings['location_name'], str(plot_freq)+'-hrs', 'Thermodynamics', 'OLR-Lightning', 'T+'+str(fclt[0])+'-to-'+str(fclt[1]))

        try:
            pngfile = plotOLRPrecipLgtng(pstart, pend, data2plot, bbox, plot_fname, settings)
            if os.path.isfile(pngfile):
                ofiles.append(pngfile)
        except:
            continue

    return ofiles

def plot_low_level(start, end, bbox, ofiles, settings):
    '''
    Postage stamp plots, columns = first 3 forecast lead times, rows = 3-hrly timesteps
    :param start: datetime. Start of the case study
    :param end: datetime. End of the case study
    :param bbox: list of [xmin, ymin, xmax, ymax]
    :param ofiles: list of output files
    :param settings: general configuration settings
    :return: list of files
    '''

    varlist = ['Uwind10m', 'Vwind10m', 'temp1.5m', 'rh1.5m']

    fchrs = sf.get_fc_InitHours(settings['jobid'])
    if not start.hour in fchrs or not end.hour in fchrs:
        start = start.replace(hour=fchrs[0])
        end = end.replace(hour=fchrs[0])

    # Most jobs run 12-hourly, either on 0 and 12Z or 6 and 18Z
    plot_freq = 3
    plot_time_range = sf.get_time_segments(start, end, 12, plot_freq, start_hour_zero=False)
    fclts = [(12,24), (24,36), (36,48)] # [(6, 18), (12,24), (18, 30), (24,36), (30,42)]
    # TODO make retrieving the model ids more generic
    model_ids = ['africa-prods', 'global-prods']
    # model_ids = [x for x in sf.getModels_bybox(bbox)['model_list']]

    for (pstart, pend), modid in itertools.product(plot_time_range, model_ids):

        print(pstart, pend, modid, fclts)

        # Get 3-hr time periods to cycle through
        row_time_ranges = sf.get_time_segments(pstart, pend, 3)

        # Make an output dictionary for sending to the plotting routine
        data2plot = {}

        for rstart, rend in row_time_ranges:

            data2plot[rend] = {}

            # Load UM data
            for fclt in fclts:
                fclt_name = 'T+' + str(fclt[0]) + '-' + str(fclt[1])
                data2plot[rend][fclt_name] = {}
                for v in varlist:
                    print(rend, fclt, v)
                    data2plot[rend][fclt_name][v] = load_data.unified_model(rstart, rend, settings, bbox=bbox,
                                                        region_type='event', model_id=[modid], var=v, fclt_clip=fclt,
                                                        aggregate=False, timeclip=True)[modid][v][0][-1, ...]

        # Do the plotting for each
        # region_name, location_name, validtime, modelid, plot_location, timeagg, plottype, plotname, fclt, outtype='filesystem'
        plot_fname = sf.make_outputplot_filename(settings['region_name'], settings['location_name'], pend,
                                                 modid, settings['location_name'], str(plot_freq) + '-hrs',
                                                 'Thermodynamics', 'Low-Level', 'All-FCLT')

        try:
            pngfile = plotLowLevel(pstart, pend, data2plot, bbox, modid, plot_fname, settings)
            if os.path.isfile(pngfile):
                ofiles.append(pngfile)
        except:
            pdb.set_trace()
            continue

    return ofiles


def get_contour_levels(data2plot, fieldname, extend='neither', level_num=200):
    '''
    Get the min and max of the field for plotting contour levels. Requires a nested dictionary with 2 levels of keys before the fieldname keys
    :param data2plot: big dictionary with 3 levels of keys
    :param fieldname: the name of the field that we need to do the stats on
    :param extend: string. Same as matplotlib, one of [ 'neither' | 'both' | 'min' | 'max' ] if 'neither', then min and max of the data will be used. If extend='min' (or 'max' or 'both') then we calculate the value of the 1st percentile instead. This removes outliers so the colour scale has more variation.
    :param level_num: a higher number makes the colorbar look more continuous
    :return: an evenly array of numbers to use as contour levels
    '''

    cubelist = iris.cube.CubeList([])
    for k1 in data2plot.keys():
        for k2 in data2plot[k1].keys():
            cube = data2plot[k1][k2][fieldname]
            cubelist.append(cube)

    cubem = cubelist.merge_cube()

    if extend == 'neither':
        fmin = cubem.data.min()
        fmax = cubem.data.max()
    elif extend == 'min':
        fmin = cubem.collapsed([c.name() for c in cubem.dim_coords], iris.analysis.PERCENTILE, percent=[1]).data.data[0]
        fmax = cubem.data.max()
    elif extend == 'max':
        fmin = cubem.data.min()
        fmax = cubem.collapsed([c.name() for c in cubem.dim_coords], iris.analysis.PERCENTILE, percent=[99]).data.data[0]
    elif extend == 'both':
        fmin = cubem.collapsed([c.name() for c in cubem.dim_coords], iris.analysis.PERCENTILE, percent=[1]).data.data[0]
        fmax = cubem.collapsed([c.name() for c in cubem.dim_coords], iris.analysis.PERCENTILE, percent=[99]).data.data[0]
    else:
        fmin = cubem.data.min()
        fmax = cubem.data.max()

    contour_levels = np.linspace(fmin, fmax, level_num)

    return contour_levels


def plotLowLevel(pstart, pend, data2plot, bbox, modid, ofile, settings):
    '''
    Do the plotting with the data2plot
    :param pstart:
    :param pend:
    :param data2plot:
    :param bbox:
    :param ofile:
    :param settings:
    :return: plot filename
    '''

    print('Plotting data from: '+pstart.strftime('%Y%m%d %H:%M')+' to '+pend.strftime('%Y%m%d %H:%M'))
    # Read data2plot to setup plot dimensions
    ncols = 3
    row_dict = {}
    cnt = 1
    for k in np.arange(len(data2plot.keys())):
        row_dict[k] = np.arange(cnt, cnt + ncols)
        cnt += ncols

    nrows = len(row_dict.keys())
    fclt1_pos = np.arange(1, 20, 3) # [1, 4, 7, 10]
    fclt2_pos = np.arange(2, 21, 3) # [2, 5, 8, 11]
    fclt3_pos = np.arange(3, 22, 3) # [3, 6, 9, 12]

    nice_models = {'africa-prods': 'UM-TA4',
                   'global-prods': 'UM-Global'}

    # TODO fix the colour scale and levels
    # contour_levels = {'3-hrs': [0.0, 0.3, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 1000.0],
    #                   '6-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
    #                   '12-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
    #                   '24-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
    #                   '48-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
    #                   '72-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
    #                   '96-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0]}
    #
    # my_rgb = ['#ffffff', '#87bbeb', '#6a9bde', '#2a6eb3', '#30ca28', '#e2d942', '#f49d1b', '#e2361d', '#f565f5',
    #           '#ffffff']
    #
    # bounds = contour_levels['3-hrs']
    # norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(my_rgb))
    # my_cmap = colors.ListedColormap(my_rgb)

    tcontour_levels = get_contour_levels(data2plot, 'temp1.5m', extend='both', level_num=200)
    rhcontour_levels = get_contour_levels(data2plot, 'rh1.5m', extend='both', level_num=5)

    # For use with pcolormesh
    cmap = plt.get_cmap('coolwarm')
    norm = colors.BoundaryNorm(tcontour_levels, ncolors=cmap.N, clip=True)

    fig = plt.figure(figsize=(16, 16), dpi=100)

    # Also manually adjust the spacings which are used when creating subplots
    plt.gcf().subplots_adjust(hspace=0.1, wspace=0.05, top=0.92, bottom=0.05, left=0.22, right=0.85)

    # iterate over all row datetimes
    i = 1
    for rowdt, data in data2plot.items():

        print(rowdt)
        rowi = list(data2plot.keys()).index(rowdt)


        for dk in data.keys():

            # i = list(data.keys()).index(dk) + 1
            print(dk)
            ax = fig.add_subplot(nrows, ncols, i, projection=ccrs.PlateCarree())
            x0, y0, x1, y1 = bbox
            ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

            # Plot temperature
            tcm = iplt.pcolormesh(data[dk]['temp1.5m'], cmap=cmap, norm=norm)
            # Plot RH contours
            rhcm = iplt.contour(data[dk]['rh1.5m'], colors=['green'], levels=rhcontour_levels)
            # Use the RH line contours to place contour labels.
            ax.clabel(rhcm, colors=['green'], manual=False, inline=True, fmt=' {:.0f} '.format)
            u = data[dk]['Uwind10m']
            v = data[dk]['Vwind10m']
            plt.title(dk.replace('-', ' to '), fontsize=14)

            # Add Coastlines, Borders and Gridlines
            lakelines = cfeature.NaturalEarthFeature(
                category='physical',
                name='lakes',
                scale='10m',
                edgecolor='black',
                alpha=0.5,
                facecolor='none')
            ax.add_feature(lakelines)
            borderlines = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_0_boundary_lines_land',
                scale='50m',
                linewidth=1,
                linestyle=(0, (3, 1, 1, 1, 1, 1)),
                edgecolor='black',
                facecolor='none')
            ax.add_feature(borderlines)
            ax.coastlines(resolution='50m', color='black')

            Y, X, U, V, lw, speed_cube = uv_to_speed_dir(u, v)
            ax.streamplot(X, Y, U, V, density=(2, 1), color='k', linewidth=lw)

            gl = ax.gridlines(color="gray", alpha=0.2, draw_labels=True)

            if rowi < (nrows - 1):
                gl.bottom_labels = False
                gl.top_labels = False
            else:
                gl.top_labels = False

            if i == row_dict[rowi][0]:
                gl.right_labels = False
                ax.text(-0.13, 0.5, rowdt.strftime('%Y%m%d %H:%M'), fontsize=14, va='center', ha='right', rotation='vertical', transform=ax.transAxes)
            elif i == row_dict[rowi][ncols-1]:
                gl.left_labels = False
            else:
                gl.right_labels = False
                gl.left_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}

            i += 1

    # Make an axes to put the shared colorbar in
    try:
        colorbar_axes = plt.gcf().add_axes([0.88, 0.5, 0.025, 0.35])  # left, bottom, width, height
        colorbar = plt.colorbar(tcm, colorbar_axes, orientation='vertical', extend='max', format=' {:.0f} '.format) # va='center',
        colorbar.set_label('Temperature (K)')
    except:
        pass

    # Make another axes for the OLR colour bar
    try:
        ocolorbar_axes = plt.gcf().add_axes([0.88, 0.1, 0.025, 0.35])  # left, bottom, width, height
        ocolorbar = plt.colorbar(rhcm, ocolorbar_axes, orientation='vertical', format=' {:.0f} '.format)
        ocolorbar.set_label('Relative Humidity (%)')
    except:
        pass

    # Use daterange in the title ...
    plt.suptitle(nice_models[modid] + ': 1.5m Temperature, RH and 10m Wind\n%s to %s (UTC)' % (pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M')), fontsize=18)

    fig.savefig(ofile, bbox_inches='tight')
    plt.close(fig)

    return ofile



def main(start=None, end=None, region_name=None, location_name=None, bbox=None, model_ids=None):

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
    # region_bbox = sf.getBBox_byRegionName(sf.getModelDomain_bybox(bbox))

    # Make the start at 0000UTC of the first day and the end 0000UTC the last day
    # start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    # if not end.hour == 0 and not end.minute == 0:
    #     end = (end + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # Time aggregation periods for all plots
    # timeaggs = [3]  # 72, 96, 120

    # Set model ids to plot (by checking the available data on disk)
    # full_file_list = get_local_flist(start, end, settings, region_type='event')
    # model_ids = list(set([os.path.basename(fn).split('_')[1] for fn in full_file_list])) # ['analysis', 'ga7', 'km4p4', 'km1p5']

    # Make an empty list for storing precip png plots
    ofiles = []

    # Run plotting functions
    # 2x2 of model @24-hr fclt, @48-hr fclt, etc
    # ofiles = plot_olr_pr_lgt(start, end, bbox, ofiles, settings)
    ofiles = plot_low_level(start, end, bbox, ofiles, settings)

    html.create(ofiles)


if __name__ == '__main__':
    main()
