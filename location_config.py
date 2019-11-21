'''
Some basic paths and details that are location specific
'''

import os
import pdb

def load_location_settings(site):
    '''
    This loads settings depending on the NMS that we're in
    '''

    settings = {}
    # Assume that the org is generic unless the site variable is in the config file
    org = 'generic'

    with open('../.config', 'r') as f:
        data = f.readlines()
        for line in data:
            try:
                var = line.split('=')[0]
                val = line.split('=')[1].replace('\n', '')
                if var == 'organisation':
                    org = val
                if org == site:
                    settings[var] = val

            except:
                continue

    # Add the synop path to the datadir ... we could do more things like this if we all agree a directory structure!
    settings['synop_path'] = settings['datadir'].rstrip('/') + '/synop/'
    settings['sounding_path'] = settings['datadir'].rstrip('/') + '/upper-air/'
    settings['gpm_path'] = settings['datadir'].rstrip('/') + '/gpm/'
    settings['um_path'] = settings['datadir'].rstrip('/') + '/UM/'

    # Make sure all the directory paths exist ...
    for k in settings.keys():
        if ('path' in k) or ('dir' in k):
            if not os.path.isdir(settings[k]):
                os.makedirs(settings[k])

    return settings
