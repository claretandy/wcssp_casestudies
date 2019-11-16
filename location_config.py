'''
Some basic paths and details that are location specific
'''

import os

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
            'plot_dir'        : '/home/hmiguel/WCSSP/plots/',
            'gpm_username'    : 'gab.miro@yahoo.com'
        }
    elif site == 'BMKG':
        site_details = {
            'datadir'     : '/path/to/folder/',
            'synop_wildcard'  : '*.csv',
            'synop_frequency': 3,
            'plot_dir'       : '/path/to/folder/plots/',
            'gpm_username'   : 'somebody@bmkg.go.id'
        }
    elif site == 'MMD':
        site_details = {
            'datadir': '/path/to/folder/',
            'synop_wildcard': '*.json',
            'synop_frequency': 3,
            'plot_dir': '/path/to/folder/plots/',
            'gpm_username': 'somebody@metmalaysia.my'
        }
    elif site == 'Andy-MacBook':
        site_details = {
            'datadir': '/Users/andy/Work/WCSSP_SEA/Data/',
            'synop_wildcard': '*.json',
            'synop_frequency': 3,
            'plot_dir': '/Users/andy/Work/WCSSP_SEA/Plots/',
            'gpm_username' : 'andrew.hartley@metoffice.gov.uk'
        }
    elif site == 'UKMO':
        site_details = {
            'datadir': '/data/users/hadhy/CaseStudies/Data/',
            'synop_wildcard': '*.json',
            'synop_frequency': 3,
            'plot_dir': '/data/users/hadhy/CaseStudies/Plots/',
            'gpm_username' : 'andrew.hartley@metoffice.gov.uk'
        }
    else:
        site_details = {
            'datadir': './Data/',
            'synop_wildcard': '*.json',
            'synop_frequency': 3,
            'plot_dir': './Plots/',
            'gpm_username': 'andrew.hartley@metoffice.gov.uk'
        }

    # Add the synop path to the datadir ... we could do more things like this if we all agree a directory structure!
    site_details['synop_path'] = site_details['datadir'] + 'synop/'
    site_details['sounding_path'] = site_details['datadir'] + 'upper-air/'
    site_details['gpm_path'] = site_details['datadir'] + 'gpm/'

    # Make sure all the directory paths exist ...
    for k in site_details.keys():
        if ('path' in k) or ('dir' in k):
            if not os.path.isdir(site_details[k]):
                os.makedirs(site_details[k])

    return site_details
