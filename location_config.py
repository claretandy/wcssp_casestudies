'''
Some basic paths and details that are location specific
'''

import os

def load_location_settings(site):
    '''
    This loads settings depending on the NMS that we're in
    '''

    # Test if we have this site in the config file, if not, then use general settings (which may not work)
    orgs = []
    with open('.config', 'r') as f:
        data = f.readlines()
        for line in data:
            try:
                val = line.split('=')[1].replace('\n', '')
            except:
                continue
            if 'organisation' in line:
                orgs.append(val)

    print(site)
    site = site if site in orgs else 'generic'
    print(site)

    settings = {}
    with open('.config', 'r') as f:
        data = f.readlines()
        for line in data:
            try:
                var = line.split('=')[0]
                val = line.split('=')[1].replace('\n', '')
                if var == 'organisation' and val == site:
                    org = val
                if org == site:
                    settings[var] = val
            except:
                continue

    # Add the synop path to the datadir ... we could do more things like this if we all agree a directory structure!
    settings['synop_path'] = settings['datadir'] + 'synop/'
    settings['sounding_path'] = settings['datadir'] + 'upper-air/'
    settings['gpm_path'] = settings['datadir'] + 'gpm/'

    # Make sure all the directory paths exist ...
    for k in settings.keys():
        if ('path' in k) or ('dir' in k):
            if not os.path.isdir(settings[k]):
                os.makedirs(settings[k])

    return settings
