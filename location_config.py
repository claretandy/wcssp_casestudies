'''
Some basic paths and details that are location specific
'''


def load_location_settings(site):
    '''
    This loads data depending on the NMS that we're in
    # Points to the SYNOP data
    '''

    if site == 'PAGASA':
        site_details = {
            'datadir'         : '/home/hmiguel/WCSSP/PAGASA/',
            'synop_wildcard'  : '*.json',
            'synop_frequency' : 3,
            'plot_dir'        : '/home/hmiguel/WCSSP/plots/'
        }
    elif site == 'BMKG':
        site_details = {
            'datadir'     : '/path/to/folder/',
            'synop_wildcard'  : '*.csv',
            'synop_frequency': 3,
            'plot_dir'       : '/path/to/folder/plots/'
        }
    elif site == 'MMD':
        site_details = {
            'datadir': '/path/to/folder/',
            'synop_wildcard': '*.json',
            'synop_frequency': 3,
            'plot_dir': '/path/to/folder/plots/'
        }
    elif site == 'Andy-MacBook':
        site_details = {
            'datadir': '/Users/andy/Work/WCSSP_SEA/PAGASA/',
            'synop_wildcard': '*.json',
            'synop_frequency': 3,
            'plot_dir': '/Users/andy/Work/WCSSP_SEA/plots/'
        }
    else:
        site_details = {
            'datadir': './Data/PAGASA/synop/',
            'synop_wildcard': '*.json',
            'synop_frequency': 3,
            'plot_dir': './plots/'
        }

    # Add the synop path to the datadir ... we could do more things like this if we all agree a directory structure!
    site_details['synop_path'] = site_details['datadir'] + 'synop/'
    site_details['sounding_path'] = site_details['datadir'] + 'upper-air/'

    return site_details
