# -*- coding: utf-8 -*-
'''
Some basic paths and details that are location specific
'''

import os
import datetime as dt

def get_site_by_ip():

    import json
    from urllib import request

    url = 'http://ipinfo.io/json'
    response = request.urlopen(url)
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

def load_location_settings(site=get_site_by_ip()):
    '''
    This loads settings depending on the NMS that we're in
    '''

    settings = {}
    start_setting = False
    got_one = False
    with open('../.config', 'r') as f:
        data = f.readlines()
        for line in data:
            if line == '\n':
                continue
            else:
                var = line.split('=')[0]
                val = line.split('=')[1].replace('\n', '').split('#')[0].strip()
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

    # Make sure all the directory paths exist
    for k in settings.keys():
        if ('path' in k) or ('dir' in k):
            if not os.path.isdir(settings[k]):
                os.makedirs(settings[k])

    # Add system environment variables to settings
    try:
        settings['start'] = dt.datetime.strptime(os.environ['start'], '%Y%m%d%H%M')
    except:
        # For realtime
        settings['start'] = dt.datetime.utcnow() - dt.timedelta(days=10)

    try:
        settings['end'] = dt.datetime.strptime(os.environ['end'], '%Y%m%d%H%M')
    except:
        # For realtime
        settings['end'] = dt.datetime.utcnow()

    try:
        settings['region_name'] = os.environ['region_name']
    except:
        # For testing
        settings['region_name'] = 'SE-Asia'

    try:
        settings['location_name'] = os.environ['location_name']
    except:
        # For testing
        settings['location_name'] = 'Peninsular-Malaysia'

    try:
        domain_str = os.environ['bbox']
        settings['bbox'] = [float(x) for x in domain_str.split(',')]
    except:
        # For testing
        settings['bbox'] = [100, 0, 110, 10]

    try:
        model_ids = os.environ['model_ids']
        if ',' in model_ids:
            settings['model_ids'] = model_ids.split(',')
        else:
            settings['model_ids'] = [model_ids]
    except:
        # For testing
        settings['model_ids'] = None

    try:
        settings['stash_colname'] = os.environ['stash_colname']
    except:
        settings['stash_colname'] = 'share_region'

    try:
        settings['ftp_upload'] = True if os.environ['ftp_upload'] == 'True' else False
    except:
        settings['ftp_upload'] = False

    # Choose from ['all', 'auto', 'production', 'NRTlate', 'NRTearly']
    # 'auto' means that it chooses the best quality expected to be available
    settings['gpm_latency'] = 'auto'

    return settings
