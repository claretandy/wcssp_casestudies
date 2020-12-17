
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
3. Edit the case study details in run_all.sh (following the instructions in the top of the file) and then run it
    ```commandline
    . run_all.sh
    ```


## Develop and contributing code 

### GitHub
Before you do anything, you will first need to register for a github account, and possibly download a local client in order to clone this repository. Here are some useful links:
- github 'Getting Started' guide: https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github
- github cheatsheets: https://training.github.com/

There are several ways you can access the repository. These are described in the sections below.

### Using PyCharm
PyCharm is a python 'Integrated Development Environment' (IDE), and is available here - https://www.jetbrains.com/pycharm/. It allows you to manage github repositories and write code without needing to learn all the github command line tools. A guide on getting started with PyCharm and GitHub is available here - https://www.jetbrains.com/help/pycharm/github.html

#### Clone the repository
1. On the github webpage https://github.com/claretandy/wcssp_casestudies hit 'Clone or download', and copy the url
2. In PyCharm, goto VCS -> Git -> Clone, and input the url you have just copied
3. Set the remote. Goto VCS -> Git -> Remotes. Enter name = origin, and url = git@github.com:claretandy/wcssp_casestudies.git

#### Create a new branch
1. You first need to pull from the remote (i.e. github) the most recent version of the remote master. This refreshes your local master with the current state of the code, and therefore brings in all the changes made by others.
2. In pycharm, you can do this by changing the branch to 'master', and clicking VCS -> Git -> Pull. This is the same as running the following on the commandline:
    ```commandline
    git checkout master
    git pull --rebase origin master
    ```
3. Now that your local master is updated, you can create a new branch. In pycharm, go to 'VCS -> Git -> Branches' then select '+ New branch'. Double check that your local master is the same as online! This is the same as running the following on the commandline:
    ```commandline
    git branch <name-of-new-branch>
    ```

4. Checkout the branch you just created. In pycharm when you create a branch, it automatically checks it out. This is the same as running the following on the commandline:
    ```commandline
    git checkout <name-of-new-branch>
    ```

5. If you haven't updated your local master for a while, you will also need to merge your new branch with the origin/master. Do this by going to VCS -> Git -> Pull. Then under 'branches to merge', select origin/master. This will force your branch to match exactly the master. You can also do this on the commandline by typing:
    ```commandline
    git push --set-upstream origin <name-of-branch>
    ```
        
6. Write some code

7. Once you have made a bit of progress, you can commit back to local repository. This is a bit like saving your work. In pycharm, go to  'VCS -> Commit'. The commandline equivalent is:
    ```commandline
    git commit -a -m 'Some description of what you've done'
    ```
        
8. Once you have a functioning piece of code, you need to send your code revision back to github. Push your changes up to the remote so that others can see what you're doing on your branch (and possibly get some help). In pycharm, go to 'VCS -> Git -> Push'. On the commandline:
    ```commandline
    git push
    ```
        
9. When you're happy that your branch works and can be committed back to the remote, you can commit and push again so that everything is up to date on github. Then, on the github website, open your branch, and issue a 'pull request' on the website.


### Using the command line
1. On the github webpage https://github.com/claretandy/wcssp_casestudies hit 'Clone or download', and copy the url

2. In a terminal, cd to a directory that you want to store the code in (e.g. ~/github ), then type the follwoing code into the terminal
    ```commandline
    git clone <URL copied from step 1>
    ```

3. Create a new branch
    ```commandline
    git branch <name of a new branch>
    ```

4. Make the new branch active
    ```commandline
    git checkout <name of new branch>
    ```

5. Link the local branch to an upstream branch
    ```commandline
    git push --set-upstream origin <name of new branch>
    ```

6. The following gives you a summary of what branch is being used
    ```commandline
    git status
    ```

7. Write some code! If this is the first time, you will need to setup your conda environment. To do this, run the environment setup script:
    ```commandline
    . run_setup.sh
    ```

8. After you have written some code, commit the code to your local repository
    - Add any new files that you have created
        ```commandline
        git add <my new file>
        ```

    - Commit the changes to the local repository
        ```commandline
        git commit -a -m 'A simple description of the changes'
        ```

    - Push the local branch to an upstream branch
        ```commandline
        git push
        ```

    - Go to the github website, find your branch, and issue a pull request for Andy to merge your code with the master

9. Once you've done this, you can delete the branch on github, then create a new branch and do something else!

