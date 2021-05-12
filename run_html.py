import os, sys
import datetime as dt
import pdb
import glob
import re
import jinja2
import shutil

def parse_filename(ifiles):
    '''
    Uses the position of strings within the filename to obtain a list of variables to put in different parts of the webpage
    Example filename: 20200520T0000Z_All-Models_SINGAPORE|CHANGI-AIRPORT_Instantaneous_upper-air_T+120.png
    :param ifiles: a list of files to go into the webpage
    :return: a dictionary of variables
    '''

    dict = {}
    fnames = [os.path.basename(f) for f in ifiles]
    dict['valid'] = sorted(list(set([f.split('_')[0] for f in fnames])))
    dict['model'] = sorted(list(set([f.split('_')[1] for f in fnames])))
    # dict['location_nice'] = sorted(list(set([f.split('_')[2].replace('-', ' ').replace('|', '/') for f in fnames])))
    dict['region'] = sorted(list(set([f.split('_')[2] for f in fnames])))
    dict['timeagg'] = sorted(list(set([f.split('_')[3] for f in fnames])))
    dict['plottype'] = sorted(list(set([os.path.dirname(f).split(os.sep)[-2] for f in ifiles])))
    dict['yearmon'] = sorted(list(set([os.path.dirname(f).split(os.sep)[-1] for f in ifiles])))
    dict['plotname'] = sorted(list(set([f.split('_')[4] for f in fnames])))
    dict['fclt'] = list(set([f.split('_')[5].replace('.png', '') for f in fnames]))

    # Sort the Forecast Lead Time better
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    dict['fclt'] = sorted(dict['fclt'], key=alphanum_key)

    # Build a page title
    # Region | Location | Upper Air
    istart = os.path.dirname(ifiles[0]).split(os.sep).index('Plots')
    region_name = os.path.dirname(ifiles[0]).split(os.sep)[istart + 1].replace('-', ' ')
    location_name = os.path.dirname(ifiles[0]).split(os.sep)[istart + 2].replace('_', ', ').replace('-', ' ').title()
    plot_type = dict['plottype'][0].replace('-', ' ').title()
    page_title = region_name + ' | ' + location_name + ' | ' + plot_type

    return dict, page_title

def nicenameLUT(keys):
    # Change these nice names to change the names on the page.
    # Don't change the keys though!
    lut = {'valid': 'Valid Time',
           'model': 'Data Source',
           'region': 'Region',
           'timeagg': 'Time Aggregation',
           'plottype': 'Plot Type',
           'plotname': 'Plot Name',
           'yearmon': 'Month',
           'fclt': 'Lead Time'}

    olist = []
    for k in keys:
        olist.append(lut[k])

    return olist


def create(ifiles):
    '''
    Uses a jinja2 template file and variables derived from the filenames in ifiles to create a web page
    :param ifiles: a list of files with local paths
    :return: html, css and js files in the directory above the ifiles
    '''
    # The template file has jinja2 code in it to allow page-specific features to be set by this script
    template_filename = "templates/file_viewer_template.html"

    # For testing
    #ifiles = glob.glob('/data/users/hadhy/wcssp_casestudies/Plots/monitoring/realtime/large-scale/*.png')
    # ifiles = sorted(glob.glob('/data/users/hadhy/wcssp_casestudies/Plots/PeninsulaMalaysia/20200520_Johor/upper-air/20*.png'))

    # Set variables from the list of input files
    # These must be formatted as follows:
    # <Valid-time>_<ModelId>_<Location>_<Time-Aggregation>_<Plot-Name>_<Lead-time>.png
    # For example:
    # 20200519T0000Z_All-Models_KUALA-LUMPUR-INTERNATIONAL-AIRPORT-(KLIA)_Instantaneous_tephigram_T+120.png
    dict, page_title = parse_filename(ifiles)

    # Get the plot type
    # Assuming we only have 1 plot type per run of this script, but we can have multiple plotnames within it
    plottype = dict['plottype'][0]

    # The rendered file goes in the directory above the plotted png files
    rendered_dirs = sorted(list(set([os.path.dirname(f).replace(os.path.dirname(f).split(os.sep)[-1], '') for f in ifiles])))
    rendered_dir = rendered_dirs[0] # Let's assume we only have 1 plottype (and therefore one dir) at the moment (but possibly multiple plotnames within it)
    rendered_filename = rendered_dir + plottype + "_latest.html"

    # Get the relative path to the files (relative to the location of the rendered html page)
    ifiles_rel = [os.path.dirname(f).split(os.sep)[-1] + '/' + os.path.basename(f) for f in ifiles] # plottype + '/' +

    # pdb.set_trace()

    # Create a dictionary from ifiles_rel that has keys that will be used for the top header (usually modelid)
    imgdict = {}
    for m in dict['model']:
        imgdict.update({m: [f for f in ifiles_rel if m == f.split('_')[1]]})

    render_vars = {
        "imgs": ifiles_rel,
        "params": dict,
        "divnames": list(dict.keys()),
        "nicedivnames": nicenameLUT(dict.keys()),
        "proc_time": dt.datetime.utcnow().strftime("%H:%M %d/%m/%Y UTC"),
        "page_title": page_title
    }

    script_path = os.path.dirname(os.path.abspath(__file__))
    # template_file_path = os.path.join(script_path, template_filename)

    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(script_path))
    output_text = environment.get_template(template_filename).render(render_vars)
    # Write the output to a file
    with open(rendered_filename, "w") as result_file:
        result_file.write(output_text)

    # Copy css and js files
    shutil.copy(script_path + '/templates/animation.js', rendered_dir + 'animation.js')
    shutil.copy(script_path + '/templates/file_viewer.css', rendered_dir + 'file_viewer.css')


if __name__ == '__main__':
    ifiles = sys.argv[1]
    create(ifiles)
