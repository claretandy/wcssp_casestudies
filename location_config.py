# -*- coding: utf-8 -*-
'''
Some basic paths and details that are location specific
'''

import os
import datetime as dt
import std_functions as sf


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

def load_location_settings(config_file=None):
    '''
    This loads settings depending on the NMS that we're in
    '''
    if config_file is None:
        config_file = "../.config"

    settings = {}
    with open(config_file, 'r') as f:
        data = f.readlines()
        for line in data:
            if line == '\n' or line[0] == '#':
                continue
            else:
                var = line.split('=')[0]
                val = line.split('=')[1].replace('\n', '').split('#')[0].strip()
                settings[var] = val

    if not 'organisation' in settings.keys():
        settings['organisation'] = get_site_by_ip()

    # Add the synop path to the datadir ... we could do more things like this if we all agree a directory structure!
    settings['synop_path'] = settings['datadir'].rstrip('/') + '/synop/'
    settings['sounding_path'] = settings['datadir'].rstrip('/') + '/upper-air/'
    settings['gpm_path'] = settings['datadir'].rstrip('/') + '/gpm/'
    settings['um_path'] = settings['datadir'].rstrip('/') + '/UM/'

    # Make sure all the directory paths exist
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
        # settings['start'] = dt.datetime(2018, 1, 27)

    try:
        settings['end'] = dt.datetime.strptime(os.environ['end'], '%Y%m%d%H%M')
    except:
        # For realtime
        settings['end'] = dt.datetime.utcnow()
        # settings['end'] = dt.datetime(2018, 1, 29)

    try:
        settings['region_name'] = os.environ['region_name']
    except:
        # For testing
        settings['region_name'] = 'SE-Asia' # 'East-Africa'

    try:
        settings['location_name'] = os.environ['location_name']
    except:
        # For testing
        settings['location_name'] = 'Northern-Vietnam' # 'Lake-Victoria' # 'Peninsular-Malaysia'

    try:
        domain_str = os.environ['bbox']
        settings['bbox'] = [float(x) for x in domain_str.split(',')]
    except:
        # For testing
        settings['bbox'] = [102.5, 16.0, 110.5, 24.0] # [30.7, -3.5, 35.3, 1.1] # [100, 0, 110, 10]

    try:
        model_ids = os.environ['model_ids']
        if ',' in model_ids:
            settings['model_ids'] = model_ids.split(',')
        else:
            settings['model_ids'] = [model_ids]
    except:
        # For testing
        settings['model_ids'] = ['analysis', 'ga6', 'km4p4'] # ['analysis', 'global', 'africa'] # ['analysis', 'ga7', 'km4p4']

    try:
        settings['ftp_upload'] = True if os.environ['ftp_upload'] == 'True' else False
    except:
        settings['ftp_upload'] = False

    settings['jobid'] = sf.getJobID_byDateTime(settings['start'], domain=sf.getModelDomain_bybox(settings['bbox']))
    # Choose from ['all', 'auto', 'production', 'NRTlate', 'NRTearly']
    # 'auto' means that it chooses the best quality expected to be available
    settings['gpm_latency'] = 'auto'

    return settings
