import location_config
from unittest.mock import patch


@patch("location_config.os")
def test_load_location_settings(mock_os, tmpdir):
    file_name = str(tmpdir / "config.text")
    content = """
organisation=UKMO
scratchdir=/scratch/myusername/
datadir=/my_data_directory/wcssp_casestudies/Data/
plot_dir=/my_data_directory/wcssp_casestudies/Plots/
code_dir=/my_home_directory/my_GitHub_dir/wcssp_casestudies/
web_dir=/my_home_directory/public_html/
url_base=https://localhost/wcssp_casestudies/
synop_station_list=./SampleData/synop/BMKG/Master.csv
synop_wildcard=*.json
synop_frequency=3
gpm_username=me@email.com
ukmo_ftp_url=ftp.metoffice.gov.uk
ukmo_ftp_user=username
ukmo_ftp_pass=password
"""
    with open(file_name, "w") as stream:
        stream.write(content)

    # Environment
    mock_os.environ = {'bbox': "0,0,0,0"}

    settings = location_config.load_location_settings(file_name)
    actual = settings["bbox"]
    expected = [0, 0, 0, 0]
    assert actual == expected
