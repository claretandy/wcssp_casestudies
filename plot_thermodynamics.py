import os
import iris
import datetime as dt
import std_functions as sf
import location_config as config
import matplotlib.pyplot as plt
import metpy


def plot_olr_pr_lgt(case_start, case_end):

    case_start = dt.datetime(2019, 3, 5)
    case_end = dt.datetime(2019, 3, 8)
    ta = 1
    max_plot_freq = 1
    time_segs = sf.get_time_segments(case_start, case_end, ta, max_plot_freq)

    for start, end in time_segs:
        print(ta, mod, start, end)


def main():

    try:
        organisation = os.environ['organisation']
    except:
        organisation = config.get_site_by_ip()

    settings = config.load_location_settings(organisation)
    start_dt = settings['start']
    end_dt = settings['end']
    region_name = settings['region_name']
    location_name = settings['location_name']
    bbox = settings['bbox']

    # Load model data

    datadir = '/scratch/hadhy/ModelData/psuite42/'
    this_dt = dt.datetime(2019, 3, 5, 6)
    olr = iris.load_cube(
        datadir + this_dt.strftime('%Y%m%dT%H%MZ') + '_africa_prods_2205_0.nc')  # 20190305T0600Z_africa_prods_2205_0.nc
    precip = iris.load_cube(datadir + this_dt.strftime('%Y%m%dT%H%MZ') + '_africa_prods_4203_0.nc')
    lightning = iris.load_cube(datadir + this_dt.strftime('%Y%m%dT%H%MZ') + '_africa_prods_21104_128.nc')

    plot_olr_pr_lgt(olr, precip, lightning)


if __name__ == '__main__':
    main()
