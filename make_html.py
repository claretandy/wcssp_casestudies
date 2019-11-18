import os
import glob
from pathlib import Path
import location_config as config

def nice_names(name):

    nndict = {'seasia'  : 'SE Asia',
              'tafrica' : 'Tropical Africa',
              'philippines': 'Philippines',
              'malaysia' : 'Malaysia',
              'indonesia': 'Indonesia'}
    try:
        return nndict[name]
    except:
        return name


def create_summary_html(organisation):

    settings = config.load_location_settings(organisation)
    indir = settings['plot_dir']
    summarypage = Path(indir).as_posix() + '/index.html'

    # NB: Directory structure is as follows:
    # <plotDir>/region_or_country/eventname/[gpm|timelagged_gpm|synop|sounding|etc]
    htmlfiles = glob.glob(Path(indir).as_posix() + '/*/*/*.html')
    gpmfiles = glob.glob(os.path.dirname(summarypage) + '/*/*/gpm/gpm_30mins_current.html')
    htmlfiles.extend(gpmfiles)
    regions = [hf.replace(Path(indir).as_posix(), '').split('/')[1] for hf in htmlfiles]
    regions = sorted(list(set(regions)))

    # Create a summary html page on which to add reference to this page

    htmlpage = open(summarypage, 'w')
    htmlpage.write('<!DOCTYPE html>\n')
    htmlpage.write('<html>\n')
    htmlpage.write('<head>\n')
    htmlpage.write('<style>\n')
    htmlpage.write('table { font-family: arial, sans-serif; border-collapse: collapse; width: 100%; }\n')
    htmlpage.write('td, th { border: 1px solid #dddddd; text-align: left; padding: 8px; }\n')
    htmlpage.write('tr:nth-child(even) { background-color: #dddddd; }\n')
    htmlpage.write('</style>\n')
    htmlpage.write('</head>\n')
    htmlpage.write('<body>\n')

    htmlpage.write('<h1>List of Case Study web pages available</h1>\n')

    for reg in regions:
        htmlpage.write('<h2>'+nice_names(reg)+'</h2>\n')
        htmlpage.write('<a id="'+reg+'"></a>\n')

        reghtmlfiles = glob.glob(Path(summarypage).with_name(reg).as_posix() + '/**/*.html')
        gpmfiles = glob.glob(os.path.dirname(summarypage) + '/' + reg + '/*/gpm/gpm_30mins_current.html')
        reghtmlfiles.extend(gpmfiles)
        regbase = [x.replace(os.path.dirname(summarypage) + '/' + reg + '/', '') for x in reghtmlfiles]

        regcases = sorted(list(set([x.split('/')[0] for x in regbase])), reverse=True)

        htmlpage.write('<table>\n')
        htmlpage.write('<tr><th>Case Study Name</th><th>GPM</th><th>Models@3hrs</th><th>Models@6hrs</th><th>Models@12hrs</th><th>Models@24hrs</th></tr>\n')

        for rc in regcases:

            # get all the html files in this Region/Case combination
            allhtml = [rh for rh in reghtmlfiles if '/'+rc+'/' in rh]
            try:
                timeaggs = [os.path.basename(x).split('.')[0].split('timeagg')[1] for x in allhtml if not 'gpm' in x]
            except:
                pdb.set_trace()

            newline = '<tr><td>'+rc+'</td><td>GPM</td><td>3hrs</td><td>6hrs</td><td>12hrs</td><td>24hrs</td></tr>\n'
            # newline = '<p>' + rc + ': GPM 3hrs 6hrs 12hrs 24hrs </p>\n'

            for ta, html in zip(timeaggs, allhtml):
                insert = ' <a href="' + reg + '/' + rc + '/' + os.path.basename(html) + '">' + ta + '</a> '
                newline = newline.replace(ta, insert)

            # Get all the 30min GPM html file for this reg/case if it exists
            gpmhtmlfile = os.path.dirname(summarypage) + '/' + reg + '/' + rc + '/gpm/gpm_30mins_current.html'
            if os.path.isfile(gpmhtmlfile):
                insert = ' <a href="' + reg + '/' + rc + '/gpm/gpm_30mins_current.html' + '">GPM</a> '
                newline = newline.replace('GPM', insert)

            # Replace all cells with no data with a '-'
            newline = re.sub('<td>(GPM|[0-9]{1,}hrs)</td>', '<td>-</td>', newline)
            htmlpage.write(newline)

        htmlpage.write('</table>\n')

    htmlpage.write('</body>\n')
    htmlpage.write('</html>\n')
    htmlpage.close()

    return summarypage

def main(organisation):


    url = create_summary_html(organisation)

    print('Created: ', url)

if __name__ == '__main__':
    organisation = sys.argv[1]
    main(organisation)
