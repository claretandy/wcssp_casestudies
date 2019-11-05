% for INI, BCT, and BCC generation

% generate/modify existing xml file; change path of input files etc., 
% see Kuya Jonat's xml file for reference/guide

% download oet; set path to oet folder
% download and organize HYCOM, WOA, etc datasets
% see HYCOM_dl.m :)

clear all; clc;

% set path
f1 = 'G:\THESIS2.0\run5\run7.xml';

% INI, BCT, or BCC
makeBctBccIni('ini','nestxml',f1)

