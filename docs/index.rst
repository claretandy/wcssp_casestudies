WCSSP Case Study Documentation
=============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

A common problem in understanding NWP (Numerical Weather Prediction) data is the ability to easily compare model forecasts to observations. This software package provides solutions to this problem by providing scripts to download and then visualise model and observational data.

Features
--------
- Read in observational data provided by National Meteorological Services in South East Asia
- Download and plot GPM IMERG data in both real time and for case studies
- Download and plot UM data from an FTP source
- Download and plot tephigrams at WMO recognised upper air stations
- Analyse the large scale ascent and descent across the tropics

Installation
------------
To run the code as a user, follow these steps:

1. clone the repository on github:

    git clone git@github.com:claretandy/wcssp_casestudies.git

2. Edit run_all.sh for the case study period that you're interested in. You'll need the following information:
      - Start and end date formatted as YYYYMMDDHHMM (e.g. 202005161700)
      - Region name for the event (e.g. Luzon, Java, Terrengganu)
      - Event name (e.g. Coldsurge1-2020, TC-Kammuri)
      - Event bounding box formatted as xmin,ymin,xmax,ymax

To access the code as a developer, see the developer guide

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
