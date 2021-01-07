import os, sys
####
# Use this for running in the Met Office on SPICE ...
import matplotlib
hname = os.uname()[1]
if not hname.startswith('eld') and not hname.startswith('els') and not hname.startswith('vld'):
   matplotlib.use('Agg')
####
import location_config as config
import std_functions as sf
import iris
import downloadUM as dum
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import iris.plot as iplt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import numpy as np
import run_html as html
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import glob
import pdb

def getHorizontalData(u, v, equator=(-30,30), plev=850):

    plevs = [('pressure', np.arange(1000, 0, -50))]
    longs = [('longitude', np.arange(0, 360, 1))]
    ueq = u.intersection(latitude=equator)
    veq = v.intersection(latitude=equator)
    ueq = ueq.extract(iris.Constraint(pressure=plev))
    veq = veq.extract(iris.Constraint(pressure=plev))

    Y = np.repeat(ueq.coord('latitude').points[..., np.newaxis], ueq.shape[1], axis=1)
    X = np.repeat(ueq.coord('longitude').points[np.newaxis, ...], ueq.shape[0], axis=0)
    U = ueq.data
    V = veq.data
    speed = np.sqrt(U**2 + V**2)
    lw = 5 * speed / speed.max() # Line width

    speed_cube = ueq.copy(speed)

    return Y, X, U, V, lw, speed_cube

def getHovmollerData(u, v, w, equator=(-5,5)):

    plevs = [('pressure', np.arange(1000, 0, -50))]
    ueq = u.intersection(latitude=equator)
    veq = v.intersection(latitude=equator)
    weq = w.intersection(latitude=equator)

    ueq2 = ueq.collapsed('latitude', iris.analysis.MEAN)
    veq2 = veq.collapsed('latitude', iris.analysis.MEAN)
    weq2 = weq.collapsed('latitude', iris.analysis.MEAN)

    ueq2 = ueq2.interpolate(plevs, iris.analysis.Linear())
    veq2 = veq2.interpolate(plevs, iris.analysis.Linear())
    weq2 = weq2.interpolate(plevs, iris.analysis.Linear())
    weq2.data = weq2.data * -1000.

    # uw_speed = ueq2.copy(np.sqrt(ueq2.data ** 2 + weq2.data ** 2))

    Y = np.repeat(ueq2.coord('pressure').points[..., np.newaxis], ueq2.shape[1], axis=1)
    X = np.repeat(ueq2.coord('longitude').points[np.newaxis, ...], ueq2.shape[0], axis=0)
    U = ueq2.data
    V = veq2.data
    W = weq2.data
    speed = np.sqrt(U ** 2 + V ** 2)
    speedw = np.sqrt(U ** 2 + W ** 2)
    lw = 5 * speed / speed.max()  # Line width
    lww = 3 * speedw / speedw.max()  # Line width

    return Y, X, U, V, W, lw, lww, ueq2

def getLandFraction(equator):

    landfrac = iris.load_cube('SampleData/um/landfrac')
    leq = landfrac.intersection(latitude=equator)
    leq2 = leq.collapsed('latitude', iris.analysis.MEAN)
    longs = [('longitude', np.arange(0, 360, 1))]
    leq2 = leq2.interpolate(longs, iris.analysis.Linear())
    leq3 = np.vstack((leq2.data, leq2.data))

    return leq3

def plot_walker(u, v, w, ofile, lats=(-5,5)):

    # Check the directory of ofile exists
    if not os.path.isdir(os.path.dirname(ofile)):
        os.makedirs(os.path.dirname(ofile))

    # Retrieve the datetime from the ofile name so that we can make a title
    this_dt = dt.datetime.strptime(os.path.basename(ofile).split('_')[0], '%Y%m%dT%H%MZ')

    os.path.basename(ofile)
    leq3 = getLandFraction(lats)
    Y, X, U, V, W, lw, lww, ueq2 = getHovmollerData(u, v, w, equator=lats)
    Yh, Xh, Uh, Vh, lwh, spdh = getHorizontalData(u, v, equator=(-30,30))

    x_tick_labels = [u'0\N{DEGREE SIGN}E', u'30\N{DEGREE SIGN}E', u'60\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}E',
                     u'120\N{DEGREE SIGN}E', u'150\N{DEGREE SIGN}E', u'180\N{DEGREE SIGN}E', u'150\N{DEGREE SIGN}W',
                     u'120\N{DEGREE SIGN}W', u'90\N{DEGREE SIGN}W', u'60\N{DEGREE SIGN}W', u'30\N{DEGREE SIGN}W',
                     u'0\N{DEGREE SIGN}W']

    fig = plt.figure(figsize=(15, 9)) # width, height
    # plt.figtext(x=0.98, y=0.96, s='Valid: ' + this_dt.strftime('%H%MUTC on %d %b %Y'), figure=fig, fontsize=16, ha='right', color='gray')
    fig.suptitle('Valid: ' + this_dt.strftime('%H%MUTC on %d %b %Y'), fontsize=14, x=0.82, ha='right', color='gray')
    plt.figtext(x=0.08, y=0.965, s='Data source: Operational UM analysis (T+0)', fontsize=14, ha='left')

    gs = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[5,3,0.5])
    crs = ccrs.PlateCarree()

    # Hovmoller of vertical profile along the tropics
    ax1 = fig.add_subplot(gs[0])
    lat0 = str(abs(lats[0]))+'S' if lats[0] < 0 else str(abs(lats[0]))+'N'
    lat1 = str(abs(lats[1])) + 'S' if lats[1] < 0 else str(abs(lats[1])) + 'N'
    ax1.set_title('Zonal and Vertical Wind for '+lat0+' to '+lat1+' (+Westerly, -Easterly)')

    uplot = ax1.contourf(X, Y, ueq2.data, cmap='RdBu_r', levels=np.arange(-15, 17, 2), extend='both')
    ax1.streamplot(X, Y, U, W, density=(5, 1), color='k', linewidth=lww)
    ax1.set_ylim((1000, 100))
    ax1.set_ylabel('Pressure Levels (hPa)')

    ax1.set_xlim((0,360))
    ax1.set_xticks(np.arange(0, 390, 30))
    ax1.set_xticklabels(x_tick_labels)
    ax1.tick_params(axis='both', labelsize=8)
    # Colorbar
    cbar1 = fig.colorbar(uplot, ax=ax1, aspect=10)
    cbar1.set_label('U component of wind (m s-1)')
    cbar1.ax.tick_params(labelsize=8)

    # 30S to 30N winds at 850hPa
    ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree(central_longitude=180))
    vplot = iplt.contourf(spdh, ax2, cmap='RdPu', levels=np.arange(0, 17, 2), extend='max')
    ax2.streamplot(Xh, Yh, Uh, Vh, density=(2,1), color='k', linewidth=lwh, transform=crs)
    ax2.set_title('Horizontal wind speed at 850hPa')
    ax2.coastlines(resolution='110m', color='white')
    gl2 = ax2.gridlines(draw_labels=True, color="gray", alpha=0.6)
    gl2.xlocator = mticker.FixedLocator([-150,-120,-90,-60,-30,0,30,60,90,120,150,180])
    gl2.ylocator = mticker.FixedLocator([-25, -15, -5, 5, 15, 25])
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER
    gl2.xlabel_style = {'size': 8}
    gl2.ylabel_style = {'size': 8}
    gl2.xlabels_top = False
    # Colorbar
    cbar2 = fig.colorbar(vplot, ax=ax2, aspect=6)
    cbar2.set_label('Wind Speed (m s-1)')
    cbar2.ax.tick_params(labelsize=8)

    # vleft, vbottom, vwidth, vheight = ax2.get_position().bounds
    # print(vheight / (1 - (vleft + vwidth)))

    # Land-sea fraction between 5S to 5N
    ax3 = fig.add_subplot(gs[2])
    lplot = ax3.pcolormesh(leq3, cmap='BrBG_r', vmin=0, vmax=1.)
    ax3.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the left edge are off
        right=False,  # ticks along the right edge are off
        labelleft=False)  # labels along the left edge are off
    ax3.set_xticks(np.arange(0, 390, 30))
    ax3.set_xticklabels(x_tick_labels)
    ax3.tick_params(axis='both', labelsize=8)
    ax3.set_title('Land Fraction')
    cbar3 = fig.colorbar(lplot, ax=ax3, aspect=1, extend='neither', ticks=[0,1])
    cbar3.ax.set_yticklabels(['Sea','Land'])
    cbar3.ax.tick_params(labelsize=8)

    # vleft, vbottom, vwidth, vheight = ax3.get_position().bounds
    # print(vheight / (1 - (vleft + vwidth)))

    # plt.tight_layout()
    # plt.show()

    fig.savefig(ofile, bbox_inches='tight')
    plt.close(fig)


def make_symlinks(ofiles):
    '''
    Makes symbolic links to the most recent files in the list of input files
    :param ofiles: list of fullpaths to plot files
    :return: creates symbolic links in the same location as the files
    '''

    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    current_files = []
    while not current_files:
        current_files = [f for f in ofiles if now.strftime('%Y%m%dT%H%MZ') in f]
        now = now - dt.timedelta(hours=1)

    for cf in current_files:
        most_recent = os.path.basename(cf).split('_')[0]
        symfile = cf.replace(most_recent, 'current')
        try:
            os.remove(symfile)
        except:
            pass
        os.symlink(cf, symfile)


def main(start, end, model_ids, event_name, organisation):
    '''
    Runs code to plot the large scale tropical circulation using the UM analysis
    :param start: datetime. Event start
    :param end: datetime. Event end
    :param model_ids: list. Could include 'analysis' or 'opfc'
    :param event_name: string. e.g. 'monitoring/realtime'
    :param organisation: 'UKMO' or other
    :return: png files in the plot directory for the event_name
    '''

    settings = config.load_location_settings(organisation)
    analysis_incr = 6
    ofiles = []
    lat_ranges = [(-5,5), (5,15), (-10,10)]

    for model_id in model_ids:
        # TODO This only works in the Met Office with analysis data at the moment, so I'll need to replace this with something like:
        # dum.loadUM
        if model_id == 'analysis':
            u_files = sf.selectAnalysisDataFromMass(start, end, 15201, lbproc=0, lblev=True)
            v_files = sf.selectAnalysisDataFromMass(start, end, 15202, lbproc=0, lblev=True)
            w_files = sf.selectAnalysisDataFromMass(start, end, 15242, lbproc=0, lblev=True)
        else:
            # This will only work for the global operational forecast (model_id = 'opfc')
            init_times = sf.getInitTimes(start, end, 'Global', model_id=model_id)
            u_files = sf.selectModelDataFromMASS(init_times, 15201, lbproc=0, lblev=True, plotdomain=[-180,-90,180,90], searchtxt=model_id)
            v_files = sf.selectModelDataFromMASS(init_times, 15202, lbproc=0, lblev=True, plotdomain=[-180, -90, 180, 90], searchtxt=model_id)
            w_files = sf.selectModelDataFromMASS(init_times, 15242, lbproc=0, lblev=True, plotdomain=[-180, -90, 180, 90], searchtxt=model_id)

        analysis_datetimes = sf.make_timeseries(start, end, analysis_incr)

        for this_dt in analysis_datetimes:

            # Format this datetime
            this_dt_fmt = this_dt.strftime('%Y%m%dT%H%MZ')

            # Subset filelists for this_dt
            try:
                u_file, = [fn for fn in u_files if this_dt_fmt in fn]
                v_file, = [fn for fn in v_files if this_dt_fmt in fn]
                w_file, = [fn for fn in w_files if this_dt_fmt in fn]
            except:
                continue

            # Make sure we have files for each variable, if we do, then load them and run the plotting code
            if u_file and v_file and w_file:
                print('Walker Circulation plotting:',this_dt_fmt)
                u = iris.load_cube(u_file)
                v = iris.load_cube(v_file)
                w = iris.load_cube(w_file)

                for lats in lat_ranges:

                    # Make nice strings of the lat min and max
                    lat0 = str(abs(lats[0])) + 'S' if lats[0] < 0 else str(abs(lats[0])) + 'N'
                    lat1 = str(abs(lats[1])) + 'S' if lats[1] < 0 else str(abs(lats[1])) + 'N'

                    # Set the output file
                    # ofile = settings['plot_dir'] + event_name + '/walker_tropics/' + this_dt_fmt + '_' + model_id + '_walker.png'
                    # <Valid-time>_<ModelId>_<Location>_<Time-Aggregation>_<Plot-Name>_<Lead-time>.png
                    ofile = sf.make_outputplot_filename(event_name, this_dt_fmt, model_id, 'Tropics-'+lat0+'-to-'+lat1,
                                                        'Instantaneous', 'large-scale', 'walker-circulation', 'T+0')

                    try:
                        if not os.path.isfile(ofile):
                            plot_walker(u, v, w, ofile, lats=lats)
                        # Append to list of ofiles
                        ofiles.append(ofile)
                    except:
                        continue

    # Make symbolic link to most recent files in ofiles
    make_symlinks(ofiles)

    # Make the html file so that the images can be viewed
    html.create(ofiles)

if __name__ == '__main__':

    try:
        start = dt.datetime.strptime(sys.argv[1], '%Y%m%d%H%M')
    except:
        # For testing
        start = dt.datetime.utcnow() - dt.timedelta(days=10)

    try:
        end = dt.datetime.strptime(sys.argv[2], '%Y%m%d%H%M')
    except:
        # For testing
        end = dt.datetime.utcnow()

    try:
        model_ids = sys.argv[3]
        model_ids = [x for x in model_ids.split(',')]
    except:
        # For testing (global bbox because we're looking at the global context of a particular case study)
        model_ids = ['analysis']

    try:
        event_name = sys.argv[4]
    except:
        # For testing
        event_name = 'monitoring/realtime'

    try:
        organisation = sys.argv[5]
    except:
        organisation = 'UKMO'

    main(start, end, model_ids, event_name, organisation)
