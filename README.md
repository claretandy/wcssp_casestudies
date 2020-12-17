
# Case Study Plotting Code

## Summary
Collaborative code to process, plot and visualise comparisons between models and observations for high impact weather case studies. This code will produce plots that can be used in reports and presentations to allow quick and easy assessment of what occurred, and how models performed.

## What does it do?
This code will help you to download, process and plot freely available observations and 'research grade' model data from a variety of sources. Current capability includes:
- pre-agreed UKMO Unified Model output (Convection permitting, global model and operational analysis) for specific case studies, 
- GPM IMERG
- radiosonde observations (via WMO and Wyoming web services)

This tool will allow you to download, organise, post-process and plot data from all of the above sources. It also provides a simple javascript-based webviewer to visualise the resulting plots.


## Running the code
If you do not want to develop or write any code, you can simply follow these steps:
1. Download the latest release from here - https://github.com/claretandy/wcssp_casestudies/releases
2. Run the environment setup script:
    ```commandline
    . run_setup.sh
    ```
3. Edit the case study details in run_all.sh and then run it


## Develop and contributing code 

### GitHub
Before you do anything, you will first need to register for a github account, and download a local client in order to clone this repository. Here are some useful links:
- github 'Getting Started' guide: https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github
- github cheatsheets: https://training.github.com/

### Using PyCharm
PyCharm is a python 'Integrated Development Environment' (IDE), and is available here - https://www.jetbrains.com/pycharm/. It allows you to manage github repositories and write code without needing to learn all the github command line tools. A guide on getting started with PyCharm and GitHub is available here - https://www.jetbrains.com/help/pycharm/github.html

   
