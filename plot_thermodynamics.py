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


def plotOLRPrecipLgtng(pstart, pend, data2plot, bbox, glomod, regmod, ofile, precip=False, lightning=True):
    '''
    Do the plotting of the data2plot
    :param pstart:
    :param pend:
    :param data2plot:
    :param bbox:
    :param glomod: name of the global model in data2plot
    :param regmod: name of the regional model in data2plot
    :param ofile:
    :param precip: boolean. If True, will plot precip and OLR
    :param lightning: boolean. If True, will plot lightning and OLR
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
    obs_pos = np.arange(1, 20, ncols) # [1, 4, 7, 10]
    reg_pos = np.arange(2, 21, ncols) # [2, 5, 8, 11]
    glo_pos = np.arange(3, 22, ncols) # [3, 6, 9, 12]

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
    pr_norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(my_rgb))
    pr_cmap = colors.ListedColormap(my_rgb)

    # Make nice model names for titles
    nice_models = {'africa-prods': 'UM-TA4',
                   'global-prods': 'UM-Global'}

    # Get contour levels that are consistent for all input data
    olr_contour_levels = sf.get_contour_levels(data2plot, 'olr', extend='both', level_num=200)
    lgt_contour_levels = sf.get_contour_levels(data2plot, 'lightning', extend='max', level_num=5)

    # For use with pcolormesh
    olr_cmap = plt.get_cmap('Greys')
    olr_norm = colors.BoundaryNorm(olr_contour_levels, ncolors=olr_cmap.N, clip=True)
    lgt_cmap = plt.get_cmap('viridis')
    lgt_norm = colors.BoundaryNorm(lgt_contour_levels, ncolors=lgt_cmap.N, clip=True)

    # Make empty colorbar objects
    pcm, ocm, lcm, lgtcm = [None, None, None, None]

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
                if not data2plot[rowdt]['obs']['olr'] == {}:
                    ocm = iplt.pcolormesh(data2plot[rowdt]['obs']['olr'], cmap=olr_cmap, norm=olr_norm)
                if precip and not data2plot[rowdt]['obs']['gpm'] == {}:
                    pcm = iplt.pcolormesh(data2plot[rowdt]['obs']['gpm'], norm=pr_norm, cmap=pr_cmap)
                if lightning and not data2plot[rowdt]['obs']['lightning'] == {}:
                    lcm = iplt.pcolormesh(data2plot[rowdt]['obs']['lightning'], norm=lgt_norm, cmap=lgt_cmap)
                plt.title('Observations', fontsize=14)

            if i in reg_pos:
                if not data2plot[rowdt][regmod]['precip'] == {}:
                    fclt = data2plot[rowdt][regmod]['precip'].coord('forecast_period').bounds[0]
                    regtitle = 'UM-TA4: T+' + str(int(fclt[0])) + ' to ' + 'T+' + str(int(fclt[1]))
                    ocm = iplt.pcolormesh(data2plot[rowdt][regmod]['olr'], cmap=olr_cmap, norm=olr_norm)
                else:
                    regtitle = ''
                if lightning and not data2plot[rowdt][regmod]['lightning'] == {}:
                    lcm = iplt.pcolormesh(data2plot[rowdt][regmod]['lightning'], norm=lgt_norm, cmap=lgt_cmap)
                if precip and not data2plot[rowdt][regmod]['precip'] == {}:
                    pcm = iplt.pcolormesh(data2plot[rowdt][regmod]['precip'], norm=pr_norm, cmap=pr_cmap)
                    # if not np.all(np.equal(lgt_contour_levels, 0)):
                    #     lgtcm = iplt.contour(data2plot[rowdt]['africa-prods']['lightning'], cmap=lgt_cmap, levels=lgt_contour_levels)
                plt.title(regtitle, fontsize=14)

            if i in glo_pos:
                if not data2plot[rowdt][glomod]['precip'] == {}:
                    fclt = data2plot[rowdt][glomod]['precip'].coord('forecast_period').bounds[0]
                    globtitle = 'UM-Global: T+' + str(int(fclt[0])) + ' to ' + 'T+' + str(int(fclt[1]))
                    ocm = iplt.pcolormesh(data2plot[rowdt][glomod]['olr'], cmap=olr_cmap, norm=olr_norm)
                else:
                    globtitle = ''
                if precip and not data2plot[rowdt][glomod]['precip'] == {}:
                    pcm = iplt.pcolormesh(data2plot[rowdt][glomod]['precip'], norm=pr_norm, cmap=pr_cmap)
                plt.title(globtitle, fontsize=14)

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
    if pcm:
        try:
            colorbar_axes = plt.gcf().add_axes([0.88, 0.65, 0.025, 0.25])  # left, bottom, width, height
            colorbar = plt.colorbar(pcm, colorbar_axes, orientation='vertical', extend='max') # va='center',
            colorbar.set_label('Precipitation accumulation (mm/3-hr)')
        except:
            pass

    # Make another axes for the OLR colour bar
    if ocm:
        try:
            ocolorbar_axes = plt.gcf().add_axes([0.88, 0.35, 0.025, 0.25])  # left, bottom, width, height
            ocolorbar = plt.colorbar(ocm, ocolorbar_axes, orientation='vertical')
            ocolorbar.set_label('Outgoing Longwave Radiation (Wm-2)')
        except:
            pass

    # Make another axes for the Lightning colour bar
    if lcm:
        try:
            ocolorbar_axes = plt.gcf().add_axes([0.88, 0.15, 0.025, 0.15])  # left, bottom, width, height
            ocolorbar = plt.colorbar(lcm, ocolorbar_axes, orientation='vertical')
            ocolorbar.set_label('Lightning flash rate (flashes/3-hr)')
        except:
            pass
    else:
        if lgtcm and not np.all(np.equal(lgt_contour_levels, 0)):
            try:
                lcolorbar_axes = plt.gcf().add_axes([0.88, 0.15, 0.025, 0.15])  # left, bottom, width, height
                lcolorbar = plt.colorbar(lgtcm, lcolorbar_axes, orientation='vertical')
                lcolorbar.set_label('Lightning flash rate (flashes/3-hr)')
            except:
                pass

    # Use daterange in the title ...
    if precip:
        title = 'OLR and Precipitation: Observations, UM-TA4 and UM-Global\n%s to %s (UTC)' % (pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M'))
    if lightning:
        title = 'OLR and Lightning: Observations, UM-TA4 and UM-Global\n%s to %s (UTC)' % (pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M'))
    if precip and lightning:
        title = 'OLR, Precipitation and Lightning: Observations, UM-TA4 and UM-Global\n%s to %s (UTC)' % (pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M'))

    plt.suptitle(title, fontsize=18)

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
    if not start.hour in fchrs:
        while not start.hour in fchrs:
            start = start - dt.timedelta(hours=1)
    if not end.hour in fchrs:
        while not end.hour in fchrs:
            end = end + dt.timedelta(hours=1)

    # Most jobs run 12-hourly, either on 0 and 12Z or 6 and 18Z
    plot_freq = 12
    plot_time_range = sf.get_time_segments(start, end, 12, plot_freq, start_hour_zero=False)
    fclts = [(9, 24), (21, 36), (33, 48)] # [(12,24), (24,36), (36,48)] # [(6, 18), (12,24), (18, 30), (24,36), (30,42)]

    # Get model ids that we can use for each column. E.g.
    # model_ids = ['africa-prods', 'global-prods']
    # glom = 'global-prods'
    # regmod = 'africa-prods'
    model_ids = sf.getModelID_byDatetime_and_bbox(start, bbox)['model_list']
    model_ids = [m.replace('_', '-') for m in model_ids]
    glomod = sf.getOneModelName(model_ids, modtype='global')
    regmod = sf.getOneModelName(model_ids, modtype='regional')

    for (pstart, pend), fclt in itertools.product(plot_time_range, fclts):

        print(pstart, pend, fclt)

        # Get 3-hr time periods to cycle through
        row_time_ranges = sf.get_time_segments(pstart, pend, 3)

        # Make an output dictionary for sending to the plotting routine
        data2plot = {}

        for rstart, rend in row_time_ranges:

            data2plot[rend] = {}
            data2plot[rend]['obs'] = {}

            # Load the UM Regional Model data
            # Do this first so that we get the grid for lightning observations
            data2plot[rend][regmod] = {}
            try:
                print('Loading regional model', regmod)
                data2plot[rend][regmod]['olr'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[regmod], var=['olr-toa'],
                                                  fclt_clip=fclt, aggregate=False, timeclip=True)[regmod]['olr-toa'][0] # [-1,...]
                data2plot[rend][regmod]['precip'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[regmod], var=['precip'],
                                                  fclt_clip=fclt, aggregate=True, totals=True, timeclip=True)[regmod]['precip']
                data2plot[rend][regmod]['lightning'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[regmod], var=['num-of-lightning-flashes'],
                                                  fclt_clip=fclt, aggregate=True, totals=True, timeclip=True)[regmod]['num-of-lightning-flashes']
            except:
                continue

            # Load observations

            # Load the GPM data
            # Aggregated for the period of the row (normally 3 hours)
            try:
                print('Loading GPM')
                data2plot[rend]['obs']['gpm'] = load_data.gpm_imerg(rstart, rend, settings, bbox=bbox, aggregate=True)
            except:
                continue

            # Load OLR Observations
            # Instantaneous obs at the end of the period of the row
            # TODO Add the satellite OLR data to the extract code (maybe a new extractObs.py script?) and FTP upload
            try:
                print('Loading OLR obs')
                data2plot[rend]['obs']['olr'] = load_data.satellite_olr(rstart, rend, settings, bbox=bbox, aggregate=False, timeseries=False)
            except:
                print("Problem loading satellite OLR data")
                continue

            # TODO Load the lightning obs if available
            try:
                print('Loading Lightning obs')
                # NB: Need a model cube to resample the lightning point obs on to
                icube = data2plot[rend][regmod]['lightning'][0]
                data2plot[rend]['obs']['lightning'] = load_data.lightning_earthnetworks(rstart, rend, icube, settings)
            except:
                print("Problem loading satellite Lightning data")
                continue

            # Load the UM Global Model data
            # glomod = 'global-prods'
            data2plot[rend][glomod] = {}
            try:
                print("Loading global model", glomod)
                data2plot[rend][glomod]['olr'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[glomod], var=['olr-toa'],
                                                  fclt_clip=fclt, aggregate=False, timeclip=True)[glomod]['olr-toa'][0] # [-1,...]
                data2plot[rend][glomod]['precip'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[glomod], var=['precip'],
                                                  fclt_clip=fclt, aggregate=True, totals=True, timeclip=True)[glomod]['precip']
                data2plot[rend][glomod]['lightning'] = load_data.unified_model(rstart, rend, settings, bbox=bbox, region_type='event',
                                                  model_id=[glomod], var=['num-of-lightning-flashes'],
                                                  fclt_clip=fclt, aggregate=True, totals=True, timeclip=True)[glomod]['num-of-lightning-flashes']
            except:
                continue

            print('Sorting out data2plot ...')
            for k in data2plot[rend].keys():
                for k2 in data2plot[rend][k].keys():
                    if data2plot[rend][k][k2] == []:
                        data2plot[rend][k][k2] = None
                    if isinstance(data2plot[rend][k][k2], list) or isinstance(data2plot[rend][k][k2], iris.cube.CubeList):
                        data2plot[rend][k][k2] = data2plot[rend][k][k2][0]
                    if 'precip' in k2:
                        data2plot[rend][k][k2].data = ma.masked_less(data2plot[rend][k][k2].data, 0.6)
                    if 'lightning' in k2 and not data2plot[rend][k][k2] == None:
                        data2plot[rend][k][k2].data = ma.masked_equal(data2plot[rend][k][k2].data, 0)
                    # if 'olr' in k2 and not data2plot[rend][k][k2] == None:
                    #     pdb.set_trace()

        # Do the plotting for OLR + Lightning
        try:
            plot_fname = sf.make_outputplot_filename(settings['region_name'], settings['location_name'], pend,
                                                     'Obs+Models', settings['location_name'], str(plot_freq) + '-hrs',
                                                     'Thermodynamics', 'OLR-Lightning',
                                                     'T+' + str(fclt[0]) + '-to-' + str(fclt[1]))
            pngfile = plotOLRPrecipLgtng(pstart, pend, data2plot, bbox, plot_fname, precip=False, lightning=True)
            if os.path.isfile(pngfile):
                ofiles.append(pngfile)
        except:
            pdb.set_trace()
            continue

        # Do the plotting for OLR + Precip
        try:
            plot_fname = sf.make_outputplot_filename(settings['region_name'], settings['location_name'], pend,
                                                     'Obs+Models', settings['location_name'], str(plot_freq) + '-hrs',
                                                     'Thermodynamics', 'OLR-Precipitation',
                                                     'T+' + str(fclt[0]) + '-to-' + str(fclt[1]))
            pngfile = plotOLRPrecipLgtng(pstart, pend, data2plot, bbox, plot_fname, precip=True, lightning=False)
            if os.path.isfile(pngfile):
                ofiles.append(pngfile)
        except:
            continue

    return ofiles


def plot_analysis_v_models(start, end, bbox, ofiles, settings, var='temp', level='near-surface'):
    '''
    TODO: Make this code work for analysis vs global vs regional model
    Postage stamp plots, columns = first 3 forecast lead times, rows = 3-hrly timesteps
    :param start: datetime. Start of the case study
    :param end: datetime. End of the case study
    :param bbox: list of [xmin, ymin, xmax, ymax]
    :param ofiles: list of output files
    :param settings: general configuration settings
    :return: list of files
    '''

    var_dict = {'near-surface': {'temp': 'temp1.5m', 'rh': 'rh1.5m', 'wind': {'U': 'Uwind10m', 'V': 'Vwind10m'}},
                850: {'temp': 'temp-levels', 'rh': 'rh-wrt-water-levels', 'wind': {'U': 'Uwind-levels', 'V': 'Vwind-levels'}},
                700: {'temp': 'temp-levels', 'rh': 'rh-wrt-water-levels', 'wind': {'U': 'Uwind-levels', 'V': 'Vwind-levels'}},
                500: {'temp': 'temp-levels', 'rh': 'rh-wrt-water-levels', 'wind': {'U': 'Uwind-levels', 'V': 'Vwind-levels'}},
                200: {'temp': 'temp-levels', 'rh': 'rh-wrt-water-levels', 'wind': {'U': 'Uwind-levels', 'V': 'Vwind-levels'}}
                }

    var_nice_names = {'temp-levels': 'Air Temperature',
                      'rh-wrt-water-levels': 'Relative Humidity',
                      'temp1.5m': '1.5m Air Temperature',
                      'rh1.5m': '1.5m Relative Humidity',
                      'Uwind10m': '10m Wind',
                      'Vwind10m': '10m Wind',
                      'Uwind-levels': 'Wind',
                      'Vwind-levels': 'Wind'
                      }

    # Sort out varlists
    uwind = var_dict[level]['wind']['U']
    vwind = var_dict[level]['wind']['V']
    grvar = var_dict[level][var]
    varlist_names = ['grvar', 'Uwind', 'Vwind']
    varlist = [grvar, uwind, vwind]

    level = level if level != 'near-surface' else None

    fchrs = sf.get_fc_InitHours(settings['jobid'])
    if not start.hour in fchrs:
        while not start.hour in fchrs:
            start = start - dt.timedelta(hours=1)
    if not end.hour in fchrs:
        while not end.hour in fchrs:
            end = end + dt.timedelta(hours=1)

    # Most jobs run 12-hourly, either on 0 and 12Z or 6 and 18Z
    rows_per_plot = 4  # How many rows of postage stamps per plot
    postage_freq = 6  # Frequency of postage plots (hours)
    plot_total_hours = postage_freq * rows_per_plot  # Time range from the start of the first plot, to the end of the last
    plot_time_range = sf.get_time_segments(start, end, plot_total_hours, postage_freq, start_hour_zero=False)
    fclts = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120)]

    # Get model ids that we can use for each column. E.g.
    # model_ids = ['africa-prods', 'global-prods']
    model_ids = sf.getModelID_byDatetime_and_bbox(start, bbox)['model_list']
    model_ids = [m.replace('_', '-') for m in model_ids]
    glomod = sf.getOneModelName(model_ids, modtype='global')
    regmod = sf.getOneModelName(model_ids, modtype='regional')

    for (pstart, pend), fclt in itertools.product(plot_time_range, fclts):

        print(pstart, pend, 'analysis', glomod, regmod, fclt)

        if level:
            plot_title = f'{var_nice_names[grvar]} and {var_nice_names[uwind]} at {level}hPa'
            var_for_filename = f'{var_nice_names[grvar]}-&-{var_nice_names[uwind]}-{level}hPa'.replace(' ','-')
        else:
            plot_title = f'{var_nice_names[grvar]} and {var_nice_names[uwind]}'
            var_for_filename = f'{var_nice_names[grvar]}-&-{var_nice_names[uwind]}'.replace(' ','-')

        # Get 3-hr time periods to cycle through
        row_time_ranges = sf.get_time_segments(pstart, pend, postage_freq)

        # Make an output dictionary for sending to the plotting routine
        data2plot = {}

        for rstart, rend in row_time_ranges:

            data2plot[rend] = {}

            # Load UM model data
            for modid in ['analysis', glomod, regmod]:

                data2plot[rend][modid] = {}
                for v in varlist:
                    vi = varlist.index(v)
                    thisfclt = None if modid == 'analysis' else fclt
                    # Get the initialisation time we will use for this plot
                    init_time = None if modid == 'analysis' else sf.get_fc_start(pstart, pend, fclt, bbox)
                    print(modid, v, rend, thisfclt)
                    try:
                        cube = load_data.unified_model(rstart, rend, settings, region_type='event', model_id=[modid], var=v, level=level, fc_init_time=init_time, aggregate=False, timeclip=True)[modid][v]

                        if isinstance(cube, list):
                            cube = cube[0]

                        if len(cube.coord('time').points) > 1:
                            data2plot[rend][modid][varlist_names[vi]] = cube[-1,...]
                        else:
                            data2plot[rend][modid][varlist_names[vi]] = cube
                    except:
                        pass

        # Do the plotting for each
        plot_fname = sf.make_outputplot_filename(settings['region_name'], settings['location_name'], pend,
                                 'All-Models', settings['location_name'], str(postage_freq) + '-hrs',
                                 'Thermodynamics', var_for_filename, 'T+' + str(fclt[0]) + '-to-' + str(fclt[1]))

        try:
            pngfile = plotAnalysisVModels(pstart, pend, data2plot, bbox, glomod, regmod, init_time, plot_fname, plot_title)
            if os.path.isfile(pngfile):
                ofiles.append(pngfile)
        except:
            break
            # continue

    return ofiles


def plotAnalysisVModels(pstart, pend, data2plot, bbox, glomod, regmod, init_time, plot_fname, plot_title):
    '''
    Plots Analysis vs Global vs Regional models for whatever variables are in data2plot
    :param pstart:
    :param pend:
    :param data2plot:
    :param bbox:
    :param glomod:
    :param regmod:
    :param init_time:
    :param plot_fname:
    :param settings:
    :return:
    '''

    print('Plotting data from: '+pstart.strftime('%Y%m%d %H:%M')+' to '+pend.strftime('%Y%m%d %H:%M'))
    # pdb.set_trace()
    # Read data2plot to setup plot dimensions
    ncols = 3
    row_dict = {}
    cnt = 1
    for k in np.arange(len(data2plot.keys())):
        row_dict[k] = np.arange(cnt, cnt + ncols)
        cnt += ncols

    # Get variable names from cubes in data2plot
    varlist = []
    for k in data2plot.keys():
        for k2 in data2plot[k].keys():
            for k3 in data2plot[k][k2].keys():
                varlist.append(data2plot[k][k2][k3].name())
    varlist = list(set(varlist))

    nrows = len(row_dict.keys())
    obs_pos = np.arange(1, 20, ncols)  # [1, 4, 7, 10]
    reg_pos = np.arange(2, 21, ncols)  # [2, 5, 8, 11]
    glo_pos = np.arange(3, 22, ncols)  # [3, 6, 9, 12]

    # Get contour levels that are consistent for all input data
    wspdcontour_levels = np.array([0.5, 2.5, 5.0, 7.5, 10.0])
    contour_levels = sf.get_contour_levels(data2plot, 'grvar', extend='both', level_num=200)
    cmap = plt.get_cmap('coolwarm') # 'viridis'
    norm = colors.BoundaryNorm(contour_levels, ncolors=cmap.N, clip=True)

    fig = plt.figure(figsize=(16, 16), dpi=100)

    # Also manually adjust the spacings which are used when creating subplots
    plt.gcf().subplots_adjust(hspace=0.1, wspace=0.05, top=0.92, bottom=0.05, left=0.22, right=0.85)

    # iterate over all row datetimes
    i = 1
    for rowdt, data in data2plot.items():

        print(rowdt)
        rowi = list(data2plot.keys()).index(rowdt)

        for dk in ['analysis', glomod, regmod]:

            print(dk)
            ax = fig.add_subplot(nrows, ncols, i, projection=ccrs.PlateCarree())
            x0, y0, x1, y1 = bbox
            ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
            if not data[dk] == {}:
                # Plot temperature
                cm = iplt.pcolormesh(data[dk]['grvar'], cmap=cmap, norm=norm)
                # Set the subplot title
                if dk == 'analysis':
                    this_title = 'UM Analysis'
                else:
                    this_fp = data[dk]['grvar'].coord('forecast_period')
                    if this_fp.has_bounds():
                        fclt = data[dk]['grvar'].coord('forecast_period').bounds[0]
                        this_title = 'T+' + str(int(fclt[0])) + ' to ' + 'T+' + str(int(fclt[1]))
                    else:
                        fclt = data[dk]['grvar'].coord('forecast_period').points[0]
                        this_title = 'T+' + str(int(fclt))
                plt.title(this_title, fontsize=14)

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

            # Plot wind streamlines
            if not data[dk] == {}:
                # pdb.set_trace()
                u = data[dk]['Uwind']
                v = data[dk]['Vwind']
                Y, X, U, V, spd, dir, lw = sf.compute_allwind(u, v, spd_levels=wspdcontour_levels)
                ax.streamplot(X, Y, U, V, density=(1, 1), color='k', linewidth=lw)

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

    # Make an axes to put the shared temperature colorbar in
    try:
        colorbar_axes = plt.gcf().add_axes([0.88, 0.2, 0.025, 0.45])  # left, bottom, width, height
        colorbar = plt.colorbar(cm, colorbar_axes, orientation='vertical', extend='max', format="%d")
        # colorbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in colorbar.ax.yaxis.get_ticklabels()])
        colorbar.set_label('Temperature (K)')
    except:
        pass

    # make an axes to put the streamlines legend in
    strax = plt.gcf().add_axes([0.89, 0.7, 0.025, 0.13])  # left, bottom, width, height
    sf.makeStreamLegend(strax, wspdcontour_levels)

    plt.suptitle(plot_title + '\n%s to %s (UTC)' % (
    pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M')), fontsize=18)

    fig.savefig(plot_fname, bbox_inches='tight')
    plt.close(fig)

    return plot_fname


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

    varlist = ['Uwind10m', 'Vwind10m', 'temp1.5m']  # 'rh1.5m'

    fchrs = sf.get_fc_InitHours(settings['jobid'])
    if not start.hour in fchrs:
        while not start.hour in fchrs:
            start = start - dt.timedelta(hours=1)
    if not end.hour in fchrs:
        while not end.hour in fchrs:
            end = end + dt.timedelta(hours=1)

    # Most jobs run 12-hourly, either on 0 and 12Z or 6 and 18Z
    plot_freq = 3
    plot_time_range = sf.get_time_segments(start, end, 12, plot_freq, start_hour_zero=False)
    fclts = [(9,24), (21,36), (33,48)] # [(9, 27), (21, 39), (33, 51)] #  # [(6, 18), (12,24), (18, 30), (24,36), (30,42)]

    # model_ids = ['africa-prods', 'global-prods']
    model_ids = sf.getModelID_byDatetime_and_bbox(start, bbox)['model_list']
    model_ids = [m.replace('_', '-') for m in model_ids if not m == 'analysis']

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
                    try:
                        cube = load_data.unified_model(rstart, rend, settings, bbox=bbox,
                                                        region_type='event', model_id=[modid], var=v, fclt_clip=fclt,
                                                        aggregate=False, timeclip=True)[modid][v][0]
                        if len(cube.coord('time').points) > 1:
                            data2plot[rend][fclt_name][v] = cube[-1,...]
                        else:
                            data2plot[rend][fclt_name][v] = cube
                    except:
                        pass

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
            break
            # continue

    return ofiles


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

    # Make nice model names for titles
    nice_models = {'africa-prods': 'UM-TA4',
                   'global-prods': 'UM-Global'}

    # tcontour_levels = sf.get_contour_levels(data2plot, 'temp1.5m', extend='both', level_num=200)
    # rhcontour_levels = sf.get_contour_levels(data2plot, 'rh1.5m', extend='both', level_num=5)
    # wspdcontour_levels = sf.get_contour_levels(data2plot, 'wind', extend='both', level_num=4)
    tcontour_levels = np.linspace(278,298,200)
    wspdcontour_levels = np.array([0.5, 2.5, 5.0, 7.5])
    # pdb.set_trace()

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

            if not data[dk] == {}:
                # Plot temperature
                tcm = iplt.pcolormesh(data[dk]['temp1.5m'], cmap=cmap, norm=norm)
                # Set the subplot title
                this_fp = data[dk]['temp1.5m'].coord('forecast_period')
                if this_fp.has_bounds():
                    fclt = data[dk]['temp1.5m'].coord('forecast_period').bounds[0]
                    plt.title('T+' + str(int(fclt[0])) + ' to ' + 'T+' + str(int(fclt[1])), fontsize=14)
                else:
                    fclt = data[dk]['temp1.5m'].coord('forecast_period').points[0]
                    plt.title('T+' + str(int(fclt)), fontsize=14)
            else:
                print('   no data to plot')

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

            # Plot wind streamlines
            if not data[dk] == {}:
                # pdb.set_trace()
                u = data[dk]['Uwind10m']
                v = data[dk]['Vwind10m']
                Y, X, U, V, spd, dir, lw = sf.compute_allwind(u, v, spd_levels=wspdcontour_levels)
                ax.streamplot(X, Y, U, V, density=(1, 1), color='k', linewidth=lw)

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

    # Make an axes to put the shared temperature colorbar in
    try:
        colorbar_axes = plt.gcf().add_axes([0.88, 0.2, 0.025, 0.45])  # left, bottom, width, height
        colorbar = plt.colorbar(tcm, colorbar_axes, orientation='vertical', extend='max', format="%d")
        # colorbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in colorbar.ax.yaxis.get_ticklabels()])
        colorbar.set_label('Temperature (K)')
    except:
        pass

    # make an axes to put the streamlines legend in
    strax = plt.gcf().add_axes([0.89, 0.7, 0.025, 0.13])  # left, bottom, width, height
    sf.makeStreamLegend(strax, wspdcontour_levels)

    # Use daterange in the title ...
    try:
        name = nice_models[modid]
    except:
        name = modid
    plt.suptitle(name + ': 1.5m Temperature and 10m Wind\n%s to %s (UTC)' % (pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M')), fontsize=18)

    fig.savefig(ofile, bbox_inches='tight')
    plt.close(fig)

    return ofile


def plotCombinedOnemodel(pstart, pend, data2plot, bbox, modid, fclt, plot_fname):

    print('Plotting data from: '+pstart.strftime('%Y%m%d %H:%M')+' to '+pend.strftime('%Y%m%d %H:%M'))
    # Read data2plot to setup plot dimensions
    ncols = 4
    row_dict = {}
    cnt = 1
    for k in np.arange(len(data2plot.keys())):
        row_dict[k] = np.arange(cnt, cnt + ncols)
        cnt += ncols

    nrows = len(row_dict.keys())
    obs_pos = np.arange(1, 20, 4)  # [1,  5,  9, 13, 17]
    kix_pos = np.arange(2, 21, 4)  # [2,  6, 10, 14, 18]
    opr_pos = np.arange(3, 22, 4)  # [3,  7, 11, 15, 19]
    olg_pos = np.arange(4, 23, 4)  # [4,  8, 12, 16, 20]

    fcltstr = 'T+' + str(int(fclt[0])) + '-' + str(int(fclt[1]))

    contour_levels = {'3-hrs': [0.0, 0.3, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 1000.0],
                      '6-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0],
                      '12-hrs': [0.0, 0.6, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 1000.0]}

    my_rgb = ['#ffffff', '#87bbeb', '#6a9bde', '#2a6eb3', '#30ca28', '#e2d942', '#f49d1b', '#e2361d', '#f565f5',
              '#ffffff']

    bounds = contour_levels['3-hrs']
    pr_norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(my_rgb))
    pr_cmap = colors.ListedColormap(my_rgb)

    # Make nice model names for titles
    nice_modelnames_lut = {'africa-prods': 'UM-TA4',
                           'global-prods': 'UM-Global'}

    # Get contour levels that are consistent for all input data
    olr_contour_levels = sf.get_contour_levels(data2plot, 'olr-toa', extend='both', level_num=200)
    lgt_contour_levels = sf.get_contour_levels(data2plot, 'num-of-lightning-flashes', extend='max', level_num=5)
    kix_contour_levels = sf.get_contour_levels(data2plot, 'k-index', extend='both', level_num=200)

    # For use with pcolormesh
    olr_cmap = plt.get_cmap('Greys')
    olr_norm = colors.BoundaryNorm(olr_contour_levels, ncolors=olr_cmap.N, clip=True)
    lgt_cmap = plt.get_cmap('viridis')
    lgt_norm = colors.BoundaryNorm(lgt_contour_levels, ncolors=lgt_cmap.N, clip=True)
    kix_cmap = plt.get_cmap('plasma')
    kix_norm = colors.BoundaryNorm(kix_contour_levels, ncolors=kix_cmap.N, clip=True)

    # Make empty colorbar objects
    pcm, ocm, lcm, lgtcm, kcm = [None, None, None, None, None]

    fig = plt.figure(figsize=(16, 13.5), dpi=150) # width, height

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

            # pdb.set_trace()

            if i in obs_pos:
                try:
                    ocm = iplt.pcolormesh(data['obs']['olr-toa'], cmap=olr_cmap, norm=olr_norm)
                except:
                    pass
                try:
                    lcm = iplt.pcolormesh(data['obs']['num-of-lightning-flashes'], norm=lgt_norm, cmap=lgt_cmap)
                except:
                    pass
                plt.title('OLR & Lightning Obs', fontsize=12)

            if i in kix_pos:
                try:
                    kcm = iplt.pcolormesh(data[fcltstr]['k-index'], cmap=kix_cmap, norm=kix_norm)
                    this_fclt = data[fcltstr]['k-index'].coord('forecast_period').points[0]
                    regtitle = 'K-Index : T+' + str(int(this_fclt))
                except:
                    regtitle = ''
                plt.title(regtitle, fontsize=12)

            if i in opr_pos:
                try:
                    ocm = iplt.pcolormesh(data[fcltstr]['olr-toa'], cmap=olr_cmap, norm=olr_norm)
                    this_fclt = data[fcltstr]['olr-toa'].coord('forecast_period').points[0]
                    regtitle = 'OLR (T+' + str(int(this_fclt)) + ')'
                except:
                    regtitle = ''
                try:
                    pcm = iplt.pcolormesh(data[fcltstr]['precip'], cmap=pr_cmap, norm=pr_norm)
                    this_fclt_bnd = data[fcltstr]['precip'].coord('forecast_period').bounds[0]
                    if regtitle == '':
                        regtitle = 'Precipitation (T+' + str(int(this_fclt_bnd[0])) + '-' + str(int(this_fclt_bnd[1])) + ')'
                    else:
                        regtitle = regtitle + ' &\nPrecipitation (T+' + str(int(this_fclt_bnd[0])) + '-' + str(int(this_fclt_bnd[1])) + ')'
                except:
                    regtitle = ''
                plt.title(regtitle, fontsize=12)

            if i in olg_pos:
                try:
                    ocm = iplt.pcolormesh(data[fcltstr]['olr-toa'], cmap=olr_cmap, norm=olr_norm)
                    this_fclt = data[fcltstr]['olr-toa'].coord('forecast_period').points[0]
                    regtitle = 'OLR (T+' + str(int(this_fclt)) + ')'
                except:
                    regtitle = ''
                try:
                    lcm = iplt.pcolormesh(data[fcltstr]['num-of-lightning-flashes'], cmap=lgt_cmap, norm=lgt_norm)
                    this_fclt_bnd = data[fcltstr]['num-of-lightning-flashes'].coord('forecast_period').bounds[0]
                    if regtitle == '':
                        regtitle = 'Lightning (T+' + str(int(this_fclt_bnd[0])) + '-' + str(int(this_fclt_bnd[1])) + ')'
                    else:
                        regtitle = regtitle + ' &\nLightning (T+' + str(int(this_fclt_bnd[0])) + '-' + str(int(this_fclt_bnd[1])) + ')'
                except:
                    regtitle = regtitle
                plt.title(regtitle, fontsize=12)

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

    # Make another axes for the Lightning colour bar
    if lcm:
        try:
            ocolorbar_axes = plt.gcf().add_axes([0.88, 0.8, 0.025, 0.12])  # left, bottom, width, height
            ocolorbar = plt.colorbar(lcm, ocolorbar_axes, orientation='vertical')
            ocolorbar.set_label('Lightning flash rate (flashes/3-hr)')
        except:
            pass
    else:
        if lgtcm and not np.all(np.equal(lgt_contour_levels, 0)):
            try:
                lcolorbar_axes = plt.gcf().add_axes([0.88, 0.8, 0.025, 0.12])  # left, bottom, width, height
                lcolorbar = plt.colorbar(lgtcm, lcolorbar_axes, orientation='vertical')
                lcolorbar.set_label('Lightning flash rate (flashes/3-hr)')
            except:
                pass

    # Make an axes to put the shared colorbar in
    if pcm:
        try:
            colorbar_axes = plt.gcf().add_axes([0.88, 0.55, 0.025, 0.18])  # left, bottom, width, height
            colorbar = plt.colorbar(pcm, colorbar_axes, orientation='vertical', extend='max') # va='center',
            colorbar.set_label('Precipitation accumulation (mm/3-hr)')
        except:
            pass

    # Make another axes for the K-Index colour bar
    if kcm:
        try:
            ocolorbar_axes = plt.gcf().add_axes([0.88, 0.3, 0.025, 0.18])  # left, bottom, width, height
            ocolorbar = plt.colorbar(kcm, ocolorbar_axes, orientation='vertical')
            ocolorbar.set_label('K-Index')
        except:
            pass

    # Make another axes for the OLR colour bar
    if ocm:
        try:
            ocolorbar_axes = plt.gcf().add_axes([0.88, 0.05, 0.025, 0.18])  # left, bottom, width, height
            ocolorbar = plt.colorbar(ocm, ocolorbar_axes, orientation='vertical')
            ocolorbar.set_label('Outgoing Longwave Radiation (Wm-2)')
        except:
            pass

    # Use daterange in the title ...
    title = 'Observations and ' + nice_modelnames_lut[modid] + '\n%s to %s (UTC)' % (pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M'))

    plt.suptitle(title, fontsize=18)

    fig.savefig(plot_fname, bbox_inches='tight')
    plt.close(fig)

    return plot_fname


def plot_combined(start, end, bbox, ofiles, settings):
    '''
    Plot some thermodynamic indices
    :param start:
    :param end:
    :param bbox:
    :param ofiles:
    :param settings:
    :return:
    '''

    varlist = ['num-of-lightning-flashes', 'precip', 'olr-toa', 'k-index']

    fchrs = sf.get_fc_InitHours(settings['jobid'])
    if not start.hour in fchrs:
        while not start.hour in fchrs:
            start = start - dt.timedelta(hours=1)
    if not end.hour in fchrs:
        while not end.hour in fchrs:
            end = end + dt.timedelta(hours=1)

    # Most jobs run 12-hourly, either on 0 and 12Z or 6 and 18Z
    plot_freq = 3
    plot_time_range = sf.get_time_segments(start, end, 12, plot_freq, start_hour_zero=False)
    fclts = [(9, 24), (21, 36), (33, 48)] # [(9, 27), (21, 39), (33, 51)] # [(6, 18), (12,24), (18, 30), (24,36), (30,42)]

    # model_ids = ['africa-prods', 'global-prods'] # No analysis because only 1 fclt available!
    model_ids = sf.getModelID_byDatetime_and_bbox(start, bbox)['model_list']
    model_ids = [m.replace('_', '-') for m in model_ids if not m == 'analysis']

    for (pstart, pend), modid in itertools.product(plot_time_range, model_ids):

        # Get 3-hr time periods to cycle through
        row_time_ranges = sf.get_time_segments(pstart, pend, 3)

        # Make an output dictionary for sending to the plotting routine
        data2plot = {}

        for rstart, rend in row_time_ranges:

            data2plot[rend] = {}
            data2plot[rend]['obs'] = {}

            # Load observations
            # Load the GPM data, aggregated for the period of the row (normally 3 hours)
            try:
                print('Loading GPM')
                data2plot[rend]['obs']['precip'] = load_data.gpm_imerg(rstart, rend, settings, bbox=bbox, aggregate=True)
            except:
                continue

            # Load OLR Observations
            # Instantaneous obs at the end of the period of the row
            # TODO Add the satellite OLR data to the extract code (maybe a new extractObs.py script?) and FTP upload
            try:
                print('Loading OLR obs')
                data2plot[rend]['obs']['olr-toa'] = load_data.satellite_olr(rstart, rend, settings, bbox=bbox, aggregate=False, timeseries=False)
            except:
                data2plot[rend]['obs']['olr-toa'] = {}
                print("Problem loading satellite OLR data")
                continue

            # TODO Load the lightning obs if available
            try:
                print('Loading Lightning obs')
                # Load some dummy data for gridding the point lightning obs
                icube = load_data.unified_model(rstart, rend, settings, bbox=bbox, fclt_clip=(12, 24), region_type='event', model_id=[modid], var='precip',aggregate=False, timeclip=True)[modid]['precip'][0]
                data2plot[rend]['obs']['num-of-lightning-flashes'] = load_data.lightning_earthnetworks(rstart, rend, icube, settings)
            except:
                data2plot[rend]['obs']['num-of-lightning-flashes'] = {}
                print("Problem loading satellite Lightning data")
                continue

            # Load UM data
            for fclt in fclts:

                fclt_name = 'T+' + str(fclt[0]) + '-' + str(fclt[1])
                fclt_name = 'T+0' if modid == 'analysis' else fclt_name
                fclt = None if modid == 'analysis' else fclt

                data2plot[rend][fclt_name] = {}

                for v in varlist:

                    print(rend, fclt, v)
                    try:
                        # OLR from the model is always at 2 mins past the hour for operational models (weird, I know!)
                        # It is also the only variable that we want to plot as an instantaneous field, rather than aggregated
                        modrend = rend if not v == 'olr-toa' else rend + dt.timedelta(minutes=15)
                        aggflag = True if not v == 'olr-toa' and not v == 'k-index' else False
                        tmp = load_data.unified_model(rstart, modrend, settings, bbox=bbox,
                            fclt_clip=fclt, region_type='event', model_id=[modid], var=v, aggregate=aggflag, timeclip=True)[modid][v][0]

                        # If we're not aggregating over the plot period, we're taking the instantaneous value, so just select the end datetime, allowing a bit of a buffer for the OLR model data
                        if not aggflag:
                            # tmp.coord('time').remove_bounds()
                            tmp = tmp.extract(iris.Constraint(time=lambda t:
                                        (rend - dt.timedelta(minutes=15)) < t.point < (rend + dt.timedelta(minutes=15))))

                        # Mask out zero, or very low values
                        if v == 'precip':
                            tmp.data = ma.masked_less(tmp.data, 0.6)
                        if v == 'num-of-lightning-flashes':
                            tmp.data = ma.masked_equal(tmp.data, 0)

                        data2plot[rend][fclt_name][v] = tmp
                    except:
                        pass

        # Do the plotting for each fclt
        # pdb.set_trace()
        for fclt in fclts:
            # region_name, location_name, validtime, modelid, plot_location, timeagg, plottype, plotname, fclt, outtype='filesystem'
            plot_fname = sf.make_outputplot_filename(settings['region_name'], settings['location_name'], pend, modid,
                                     settings['location_name'], str(plot_freq) + '-hrs', 'Thermodynamics', 'Combined-Plot', 'T+' + str(fclt[0]) + '-to-' + str(fclt[1]))

            try:
                pngfile = plotCombinedOnemodel(pstart, pend, data2plot, bbox, modid, fclt, plot_fname)
                if os.path.isfile(pngfile):
                    ofiles.append(pngfile)
            except:
                continue

    return ofiles


def plot_indices(start, end, bbox, ofiles, settings):
    '''
    Plot some thermodynamic indices
    :param start:
    :param end:
    :param bbox:
    :param ofiles:
    :param settings:
    :return:
    '''

    varlist = ['total-totals', 'k-index'] # ['lifted-index', 'total-totals', 'k-index', 'showalter-index']

    fchrs = sf.get_fc_InitHours(settings['jobid'])
    if not start.hour in fchrs:
        while not start.hour in fchrs:
            start = start - dt.timedelta(hours=1)
    if not end.hour in fchrs:
        while not end.hour in fchrs:
            end = end + dt.timedelta(hours=1)

    # Most jobs run 12-hourly, either on 0 and 12Z or 6 and 18Z
    plot_freq = 3
    plot_time_range = sf.get_time_segments(start, end, 12, plot_freq, start_hour_zero=False)
    fclts = [(9, 24), (21, 36), (33, 48)] # [(12,24), (24,36), (36,48)] # [(6, 18), (12,24), (18, 30), (24,36), (30,42)]

    # model_ids = ['africa-prods', 'global-prods'] # No analysis because only 1 fclt available!
    model_ids = sf.getModelID_byDatetime_and_bbox(start, bbox)['model_list']
    model_ids = [m.replace('_', '-') for m in model_ids if not m == 'analysis']

    for v in varlist:
        for (pstart, pend), modid in itertools.product(plot_time_range, model_ids):

            # Get 3-hr time periods to cycle through
            row_time_ranges = sf.get_time_segments(pstart, pend, 3)

            # Make an output dictionary for sending to the plotting routine
            data2plot = {}

            for rstart, rend in row_time_ranges:

                rstart = rend - dt.timedelta(hours=1)

                data2plot[rend] = {}

                # Load UM data
                for fclt in fclts:
                    fclt_name = 'T+' + str(fclt[0]) + '-' + str(fclt[1])
                    fclt_name = 'T+0' if modid == 'analysis' else fclt_name
                    data2plot[rend][fclt_name] = {}

                    print(rend, fclt, v)
                    fclt = None if modid == 'analysis' else fclt
                    try:
                        data2plot[rend][fclt_name][v] = load_data.unified_model(rstart, rend, settings, bbox=bbox,
                            fclt_clip=fclt, region_type='event', model_id=[modid], var=v, aggregate=False, timeclip=True)[modid][v][0]
                    except:
                        pass

            # Do the plotting for each
            # region_name, location_name, validtime, modelid, plot_location, timeagg, plottype, plotname, fclt, outtype='filesystem'
            plot_fname = sf.make_outputplot_filename(settings['region_name'], settings['location_name'], pend,
                                                 modid, settings['location_name'], str(plot_freq) + '-hrs',
                                                 'Thermodynamics', v, 'All-FCLT')

            try:
                pngfile = plotIndices_onemodel_allfclt(pstart, pend, data2plot, bbox, modid, v, plot_fname)
                if os.path.isfile(pngfile):
                    ofiles.append(pngfile)
            except:
                pdb.set_trace()
                continue

    return ofiles


def plotIndices_allmodels_onefclt(pstart, pend, data2plot, bbox, plot_fname):
    '''
    Plots Analysis, 4.4km, and global at 4 time slices for a given model initialisation
    :param pstart:
    :param pend:
    :param data2plot:
    :param bbox:
    :param ofile:
    :param precip: boolean. If True, will plot precip and OLR
    :param lightning: boolean. If True, will plot lightning and OLR
    :return: plot filename
    '''

    print('Plotting data from: ' + pstart.strftime('%Y%m%d %H:%M') + ' to ' + pend.strftime('%Y%m%d %H:%M'))
    # Read data2plot to setup plot dimensions
    ncols = 3
    row_dict = {}
    cnt = 1
    for k in np.arange(len(data2plot.keys())):
        row_dict[k] = np.arange(cnt, cnt + ncols)
        cnt += ncols

    nrows = len(row_dict.keys())
    obs_pos = np.arange(1, 20, 3)  # [1, 4, 7, 10]
    reg_pos = np.arange(2, 21, 3)  # [2, 5, 8, 11]
    glo_pos = np.arange(3, 22, 3)  # [3, 6, 9, 12]

    my_rgb = ['#ffffff', '#87bbeb', '#6a9bde', '#2a6eb3', '#30ca28', '#e2d942', '#f49d1b', '#e2361d', '#f565f5',
              '#ffffff']

    # LI 6 or Greater, Very Stable Conditions
    # LI Between 1 and 6 : Stable Conditions, Thunderstorms Not Likely
    # LI Between 0 and -2 : Slightly Unstable, Thunderstorms Possible, With Lifting Mechanism (i.e., cold front, daytime heating, ...)
    # LI Between -2 and -6 : Unstable, Thunderstorms Likely, Some Severe With Lifting Mechanism
    # LI Less Than -6: Very Unstable, Severe Thunderstorms Likely With Lifting Mechanism
    bounds = [-6, -2, 0, 6]
    pr_norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(my_rgb))
    pr_cmap = colors.ListedColormap(my_rgb)

    # Make nice model names for titles
    nice_models = {'analysis': 'UM-Analysis',
                   'africa-prods': 'UM-TA4',
                   'global-prods': 'UM-Global'}

    # Get contour levels that are consistent for all input data
    olr_contour_levels = sf.get_contour_levels(data2plot, 'olr', extend='both', level_num=200)
    lgt_contour_levels = sf.get_contour_levels(data2plot, 'lightning', extend='max', level_num=5)

    # For use with pcolormesh
    olr_cmap = plt.get_cmap('Greys')
    olr_norm = colors.BoundaryNorm(olr_contour_levels, ncolors=olr_cmap.N, clip=True)
    lgt_cmap = plt.get_cmap('viridis')
    lgt_norm = colors.BoundaryNorm(lgt_contour_levels, ncolors=lgt_cmap.N, clip=True)

    # Make empty colorbar objects
    pcm, ocm, lcm, lgtcm = [None, None, None, None]

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
                ocm = iplt.pcolormesh(data2plot[rowdt]['obs']['olr'], cmap=olr_cmap, norm=olr_norm)
                if precip:
                    pcm = iplt.pcolormesh(data2plot[rowdt]['obs']['gpm'], norm=pr_norm, cmap=pr_cmap)
                if lightning:
                    lcm = iplt.pcolormesh(data2plot[rowdt]['obs']['lightning'], norm=lgt_norm, cmap=lgt_cmap)
                plt.title('Observations', fontsize=14)

            if i in reg_pos:
                fclt = data2plot[rowdt]['africa-prods']['precip'].coord('forecast_period').bounds[0]
                ocm = iplt.pcolormesh(data2plot[rowdt]['africa-prods']['olr'], cmap=olr_cmap, norm=olr_norm)
                if lightning:
                    lcm = iplt.pcolormesh(data2plot[rowdt]['africa-prods']['lightning'], norm=lgt_norm,
                                          cmap=lgt_cmap)
                if precip:
                    pcm = iplt.pcolormesh(data2plot[rowdt]['africa-prods']['precip'], norm=pr_norm, cmap=pr_cmap)
                    # if not np.all(np.equal(lgt_contour_levels, 0)):
                    #     lgtcm = iplt.contour(data2plot[rowdt]['africa-prods']['lightning'], cmap=lgt_cmap, levels=lgt_contour_levels)
                plt.title('UM-TA4: T+' + str(int(fclt[0])) + ' to ' + 'T+' + str(int(fclt[1])), fontsize=14)

            if i in glo_pos:
                fclt = data2plot[rowdt]['global-prods']['precip'].coord('forecast_period').bounds[0]
                ocm = iplt.pcolormesh(data2plot[rowdt]['global-prods']['olr'], cmap=olr_cmap, norm=olr_norm)
                if precip:
                    pcm = iplt.pcolormesh(data2plot[rowdt]['global-prods']['precip'], norm=pr_norm, cmap=pr_cmap)
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
                ax.text(-0.13, 0.5, rowdt.strftime('%Y%m%d %H:%M'), fontsize=14, va='center', ha='right',
                        rotation='vertical', transform=ax.transAxes)
            elif i == row_dict[rowi][ncols - 1]:
                gl.left_labels = False
            else:
                gl.right_labels = False
                gl.left_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}

    # Make an axes to put the shared colorbar in
    if pcm:
        try:
            colorbar_axes = plt.gcf().add_axes([0.88, 0.65, 0.025, 0.25])  # left, bottom, width, height
            colorbar = plt.colorbar(pcm, colorbar_axes, orientation='vertical', extend='max')  # va='center',
            colorbar.set_label('Precipitation accumulation (mm/3-hr)')
        except:
            pass

    # Make another axes for the OLR colour bar
    if ocm:
        try:
            ocolorbar_axes = plt.gcf().add_axes([0.88, 0.35, 0.025, 0.25])  # left, bottom, width, height
            ocolorbar = plt.colorbar(ocm, ocolorbar_axes, orientation='vertical')
            ocolorbar.set_label('Outgoing Longwave Radiation (Wm-2)')
        except:
            pass

    # Make another axes for the Lightning colour bar
    if lcm:
        try:
            ocolorbar_axes = plt.gcf().add_axes([0.88, 0.15, 0.025, 0.15])  # left, bottom, width, height
            ocolorbar = plt.colorbar(lcm, ocolorbar_axes, orientation='vertical')
            ocolorbar.set_label('Lightning flash rate (flashes/3-hr)')
        except:
            pass
    else:
        if lgtcm and not np.all(np.equal(lgt_contour_levels, 0)):
            try:
                lcolorbar_axes = plt.gcf().add_axes([0.88, 0.15, 0.025, 0.15])  # left, bottom, width, height
                lcolorbar = plt.colorbar(lgtcm, lcolorbar_axes, orientation='vertical')
                lcolorbar.set_label('Lightning flash rate (flashes/3-hr)')
            except:
                pass

    # Use daterange in the title ...
    if precip:
        title = 'OLR and Precipitation: Observations, UM-TA4 and UM-Global\n%s to %s (UTC)' % (
        pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M'))
    if lightning:
        title = 'OLR and Lightning: Observations, UM-TA4 and UM-Global\n%s to %s (UTC)' % (
        pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M'))
    else:
        title = 'OLR, Precipitation and Lightning: Observations, UM-TA4 and UM-Global\n%s to %s (UTC)' % (
        pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M'))
    plt.suptitle(title, fontsize=18)

    fig.savefig(ofile, bbox_inches='tight')
    plt.close(fig)

    return ofile


def plotIndices_onemodel_allfclt(pstart, pend, data2plot, bbox, modid, indx, plot_fname):
    '''
    Plots first 3 available forecast lead times for 4 time slices for a particular model
    :param pstart:
    :param pend:
    :param data2plot:
    :param bbox:
    :param modid:
    :param plot_fname:
    :return:
    '''

    print('Plotting '+indx.replace('-', ' ')+' from: '+pstart.strftime('%Y%m%d %H:%M')+' to '+pend.strftime('%Y%m%d %H:%M'))

    # Read data2plot to setup plot dimensions
    ncols = 3
    row_dict = {}
    cnt = 1
    for k in np.arange(len(data2plot.keys())):
        row_dict[k] = np.arange(cnt, cnt + ncols)
        cnt += ncols

    nrows = len(row_dict.keys())

    # Make nice model names for titles
    nice_models = {'analysis': 'UM-Analysis',
                   'africa-prods': 'UM-TA4',
                   'global-prods': 'UM-Global'}

    levels = sf.get_contour_levels(data2plot, indx, extend='both', level_num=200)

    # For use with pcolormesh
    cmap = plt.get_cmap('viridis')
    norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig = plt.figure(figsize=(16, 16), dpi=100)

    # Also manually adjust the spacings which are used when creating subplots
    plt.gcf().subplots_adjust(hspace=0.1, wspace=0.05, top=0.92, bottom=0.05, left=0.22, right=0.85)

    # iterate over all row datetimes
    i = 1
    for rowdt, data in data2plot.items():

        print(rowdt)
        rowi = list(data2plot.keys()).index(rowdt)

        for dk in data.keys():

            print(dk)
            ax = fig.add_subplot(nrows, ncols, i, projection=ccrs.PlateCarree())
            x0, y0, x1, y1 = bbox
            ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())

            # Plot temperature
            if not data[dk] == {}:
                indx_cm = iplt.pcolormesh(data[dk][indx], cmap=cmap, norm=norm)
                # pdb.set_trace()
                # Use the RH line contours to place contour labels.
                # ax.clabel(rhcm, colors=['green'], manual=False, inline=True, fmt=' {:.0f} '.format)
                if data[dk][indx].coord('forecast_period').has_bounds():
                    fclt = data[dk][indx].coord('forecast_period').bounds[0]
                    plt.title('T+' + str(int(fclt[0])) + ' to ' + 'T+' + str(int(fclt[1])), fontsize=14)
                else:
                    fclt = data[dk][indx].coord('forecast_period').points[0]
                    plt.title('T+' + str(int(fclt)), fontsize=14)
            else:
                indx_cm = None

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

            i += 1

    # Make an axes to put the shared temperature colorbar in
    try:
        colorbar_axes = plt.gcf().add_axes([0.88, 0.5, 0.025, 0.35])  # left, bottom, width, height
        colorbar = plt.colorbar(indx_cm, colorbar_axes, orientation='vertical', extend='max', format="%d")
        # colorbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in colorbar.ax.yaxis.get_ticklabels()])
        colorbar.set_label(indx.replace('-', ' ').title())
    except:
        pass

    # Use daterange in the title ...
    plt.suptitle(nice_models[modid] + ': '+indx.replace('-', ' ').title()+'\n%s to %s (UTC)' % (pstart.strftime('%Y%m%d %H:%M'), pend.strftime('%Y%m%d %H:%M')), fontsize=18)

    fig.savefig(plot_fname, bbox_inches='tight')
    plt.close(fig)

    return plot_fname


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

    # Make an empty list for storing precip png plots
    ofiles = []

    # Run plotting functions
    ofiles = plot_combined(start, end, bbox, ofiles, settings)
    # ofiles = plot_olr_pr_lgt(start, end, bbox, ofiles, settings)
    # ofiles = plot_low_level(start, end, bbox, ofiles, settings)
    # ofiles = plot_analysis_v_models(start, end, bbox, ofiles, settings, var='temp', level='near-surface')
    ofiles = plot_analysis_v_models(start, end, bbox, ofiles, settings, var='temp', level=850)
    # ofiles = plot_analysis_v_models(start, end, bbox, ofiles, settings, var='temp', level=700)
    # ofiles = plot_analysis_v_models(start, end, bbox, ofiles, settings, var='temp', level=500)
    # ofiles = plot_analysis_v_models(start, end, bbox, ofiles, settings, var='temp', level=200)
    # ofiles = plot_indices(start, end, bbox, ofiles, settings)
    #
    html.create(ofiles)


if __name__ == '__main__':
    main()
