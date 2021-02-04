import os, sys
import pandas as pd
import datetime as dt
import subprocess
import stat


def main():

    # Reads the std_domains file and creates extract jobs for each domain (realtime and case study)
    df = pd.read_csv('std_domains.csv')

    for row in df.itertuples():

        model_ids = row.model_ids.split(',')

        for model in model_ids:

            now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            try:
                os.environ['start'] = dt.datetime.strptime(row.start, '%Y%m%d').strftime('%Y%m%d%H%M')
            except:
                os.environ['start'] = (now - dt.timedelta(days=7)).strftime('%Y%m%d%H%M')

            try:
                os.environ['end'] = dt.datetime.strptime(row.end, '%Y%m%d').strftime('%Y%m%d%H%M')
            except:
                os.environ['end'] = now.strftime('%Y%m%d%H%M')

            os.environ['region_name'] = row.region_name
            os.environ['location_name'] = row.location_name
            os.environ['bbox'] = row.bbox
            os.environ['model_ids'] = model
            os.environ['ftp_upload'] = str(row.ftp_upload)

            code_dir = os.getcwd()
            shell_script = code_dir + '/batch_output/run_extract_'+row.location_name+'_'+row.start+'_'+model+'.sh'

            with open(shell_script, 'w') as the_file:

                the_file.write('#!/bin/bash -l\n')
                the_file.write('#SBATCH --qos=long\n')
                the_file.write('#SBATCH --mem=10000\n')
                the_file.write('#SBATCH --output='+shell_script.replace('.sh', '_%j_%N.out')+'\n')
                the_file.write('#SBATCH --time=4320 \n')
                the_file.write('\n')
                the_file.write('. ~/.profile\n')
                the_file.write('export start=' + os.environ["start"] + '\n')
                the_file.write('export end=' + os.environ["end"] + '\n')
                the_file.write('export region_name=' + os.environ["region_name"] + '\n')
                the_file.write('export location_name=' + os.environ["location_name"] + '\n')
                the_file.write('export bbox=' + os.environ["bbox"] + '\n')
                the_file.write('export model_ids=' + os.environ["model_ids"] + '\n')
                the_file.write('export ftp_upload=' + os.environ["ftp_upload"] + '\n')
                the_file.write('conda activate scitools\n')
                the_file.write('\n')
                the_file.write('cd '+code_dir+'\n')
                the_file.write('echo ${region_name} ${location_name}\n')
                the_file.write('echo "Extracting: '+model+'"\n')
                the_file.write('python extractUM.py\n')

            st = os.stat(shell_script)
            os.chmod(shell_script, st.st_mode | stat.S_IEXEC)
            print(shell_script)
            subprocess.run(['sbatch', shell_script], capture_output=True)


if __name__ == '__main__':
    main()
