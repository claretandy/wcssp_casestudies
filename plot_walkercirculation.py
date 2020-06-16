import location_config as config
import iris
import downloadUM as dum
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import iris.plot as iplt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import numpy as np

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

def getHovmollerData(u, v, equator=(-5,5)):

    plevs = [('pressure', np.arange(1000, 0, -50))]
    ueq = u.intersection(latitude=equator)
    veq = v.intersection(latitude=equator)

    ueq2 = ueq.collapsed('latitude', iris.analysis.MEAN)
    veq2 = veq.collapsed('latitude', iris.analysis.MEAN)

    ueq2 = ueq2.interpolate(plevs, iris.analysis.Linear())
    veq2 = veq2.interpolate(plevs, iris.analysis.Linear())

    Y = np.repeat(ueq2.coord('pressure').points[..., np.newaxis], ueq2.shape[1], axis=1)
    X = np.repeat(ueq2.coord('longitude').points[np.newaxis, ...], ueq2.shape[0], axis=0)
    U = ueq2.data
    V = veq2.data
    speed = np.sqrt(U ** 2 + V ** 2)
    lw = 5 * speed / speed.max()  # Line width

    return Y, X, U, V, lw, ueq2

def getLandFraction(equator):

    landfrac = iris.load_cube('SampleData/um/landfrac')
    leq = landfrac.intersection(latitude=equator)
    leq2 = leq.collapsed('latitude', iris.analysis.MEAN)
    longs = [('longitude', np.arange(0, 360, 1))]
    leq2 = leq2.interpolate(longs, iris.analysis.Linear())
    leq3 = np.vstack((leq2.data, leq2.data))

    return leq3

def plot_walker(u, v, ofile):

    equator = (-5, 5)
    leq3 = getLandFraction(equator)
    Y, X, U, V, lw, ueq2 = getHovmollerData(u,v, equator=equator)
    Yh, Xh, Uh, Vh, lwh, spdh = getHorizontalData(u, v, equator=(-30,30))

    x_tick_labels = [u'0\N{DEGREE SIGN}E', u'30\N{DEGREE SIGN}E', u'60\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}E',
                     u'120\N{DEGREE SIGN}E', u'150\N{DEGREE SIGN}E', u'180\N{DEGREE SIGN}E', u'150\N{DEGREE SIGN}W',
                     u'120\N{DEGREE SIGN}W', u'90\N{DEGREE SIGN}W', u'60\N{DEGREE SIGN}W', u'30\N{DEGREE SIGN}W',
                     u'0\N{DEGREE SIGN}W']

    fig = plt.figure(figsize=(15, 7)) # width, height
    gs = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[5,3,0.5])
    crs = ccrs.PlateCarree()

    # Hovmoller of vertical profile along the tropics
    ax1 = fig.add_subplot(gs[0])
    uplot = ax1.contourf(X, Y, ueq2.data, cmap='RdBu_r', levels=np.arange(-15, 17, 2), extend='both')
    ax1.streamplot(X, Y, U, V, density=(1.2, 1.2), color='k', linewidth=lw)
    ax1.set_ylim((1000,100))
    ax1.set_ylabel('Pressure Levels (hPa)')
    ax1.set_title('Zonal Wind (+Westerly, -Easterly)')
    ax1.set_xticks(np.arange(0, 390, 30))
    ax1.set_xticklabels(x_tick_labels)
    ax1.tick_params(axis='both', labelsize=8)
    # Colorbar
    cbar1 = fig.colorbar(uplot, ax=ax1, aspect=10)
    cbar1.set_label('U component of wind (m s-1)')
    cbar1.ax.tick_params(labelsize=8)

    # vleft, vbottom, vwidth, vheight = ax1.get_position().bounds
    # print(vheight / (1 - (vleft + vwidth)))

    # 30S to 30N winds at 850hPa
    ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree(central_longitude=180))
    vplot = iplt.contourf(spdh, ax2, cmap='viridis_r', levels=np.arange(0, 17, 2), extend='max')
    ax2.streamplot(Xh, Yh, Uh, Vh, density=(1,1), color='k', linewidth=lwh, transform=crs)
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


def main(start, end, model_id, organisation):
    '''
    Runs code to plot the large scale tropical circulation using the UM analysis
    :param start:
    :param end:
    :param model_id:
    :param organisation:
    :return:
    '''

    settings = config.load_location_settings(organisation)

    # Get data
    u = iris.load_cube('/scratch/hadhy/ModelData/um_analysis/20200613T1200Z_analysis_15201_0.nc')
    v = iris.load_cube('/scratch/hadhy/ModelData/um_analysis/20200613T1200Z_analysis_15202_0.nc')
    w = iris.load_cube('/scratch/hadhy/ModelData/um_analysis/20200613T1200Z_analysis_15242_0.nc')

    plot_walker(u, v, ofile)

    # dum.loadUM(start, end, 'analysis')