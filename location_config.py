'''
Some basic paths and details that are location specific
'''

import os
import pdb

def get_site_by_ip():

    import json
    import urllib

    url = 'http://ipinfo.io/json'
    response = urllib.request.urlopen(url)
    data = json.load(response)

    country = data['country']

    site_lut = {'GB': 'UKMO',
                'PH': 'PAGASA',
                'MY': 'MMD',
                'ID': 'BMKG',
                'VN': 'Vietnam',
                'TH': 'Thailand'
                }

    try:
        return site_lut[country]
    except:
        return 'generic'

def load_location_settings(site):
    '''
    This loads settings depending on the NMS that we're in
    '''

    settings = {}
    # Assume that the org is generic unless the site variable is in the config file
    start_setting = False
    got_one = False
    with open('../.config', 'r') as f:
        data = f.readlines()
        for line in data:
            if line == '\n':
                continue
            else:
                var = line.split('=')[0]
                val = line.split('=')[1].replace('\n', '')
                if var == 'organisation':
                    # print(val)
                    if val == site:
                        got_one = True
                        start_setting = True
                    elif (not got_one) and (val == 'generic'):
                        print('Site', site, 'not available, setting generic configuration instead')
                        site = 'generic'
                        start_setting = True
                    else:
                        start_setting = False
                if start_setting:
                    # print(var, val, sep=': ')
                    settings[var] = val

    countryLUT = {
        'MMD': 'Malaysia',
        'PAGASA': 'Philippines',
        'BMKG': 'Indonesia',
        'Andy-MacBook': 'Malaysia',
        'UKMO': 'Malaysia',
        'generic': 'SEAsia'
    }

    # Add the synop path to the datadir ... we could do more things like this if we all agree a directory structure!
    settings['synop_path'] = settings['datadir'].rstrip('/') + '/synop/'
    settings['sounding_path'] = settings['datadir'].rstrip('/') + '/upper-air/'
    settings['gpm_path'] = settings['datadir'].rstrip('/') + '/gpm/'
    settings['um_path'] = settings['datadir'].rstrip('/') + '/UM/'
    settings['country'] = countryLUT[site]

    # Make sure all the directory paths exist ...
    for k in settings.keys():
        if ('path' in k) or ('dir' in k):
            if not os.path.isdir(settings[k]):
                os.makedirs(settings[k])

    return settings
