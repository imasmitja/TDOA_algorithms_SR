# -*- coding: utf-8 -*-
"""
Created on Tue Jan 07 08:47:12 2020

@author: Ivan Masmitja
"""

import numpy as np
import scipy.signal
#from scipy import io
import peakutils
import matplotlib.pyplot as plt
import time
import utm #(Ivan)
import matplotlib.path as mpltPath
import httplib, urllib
import requests
import random
import matplotlib.cm as cm
import calendar
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib

SAVE_FOLDER = 'TagDetectionsRESNEP7'


filename_BS1 = SAVE_FOLDER+r'_BS1\detections_BS1_A69-1602-65266.txt'
filename_BS2 = SAVE_FOLDER+r'_BS2\detections_BS2_A69-1602-65266.txt'
filename_BS3 = SAVE_FOLDER+r'_BS3\detections_BS3_A69-1602-65266.txt'
filename_BS4 = SAVE_FOLDER+r'_BS4\detections_BS4_A69-1602-65266.txt'
filename_BSM = SAVE_FOLDER+r'_BSM\detections_merge_A69-1602-65266.txt'

timestamp_BS1 = np.loadtxt(filename_BS1,skiprows=1, delimiter=',',usecols=(0), unpack=True)
timestamp_BS2 = np.loadtxt(filename_BS2,skiprows=1, delimiter=',',usecols=(0), unpack=True)
timestamp_BS3 = np.loadtxt(filename_BS3,skiprows=1, delimiter=',',usecols=(0), unpack=True)
timestamp_BS4 = np.loadtxt(filename_BS4,skiprows=1, delimiter=',',usecols=(0), unpack=True)
timestamp_BSM = np.loadtxt(filename_BSM,skiprows=1, delimiter=',',usecols=(0), unpack=True)

#%%

ls=12
fig = plt.figure(figsize=(5,5))

x1 = [datetime.utcfromtimestamp(element) for element in timestamp_BS1]
x2 = [datetime.utcfromtimestamp(element) for element in timestamp_BS2]
x3 = [datetime.utcfromtimestamp(element) for element in timestamp_BS3]
x4 = [datetime.utcfromtimestamp(element) for element in timestamp_BS4]
xmerge = [datetime.utcfromtimestamp(element) for element in timestamp_BSM]


plt.plot(x1, np.ones(len(x1))*1., 'bo', label="BS1") 
plt.plot(x2, np.ones(len(x2))*2., 'ro',label="BS2") 
plt.plot(x3, np.ones(len(x3))*3., 'go',label="BS3") 
plt.plot(x4, np.ones(len(x4))*4., 'yo',label="BS4") 
plt.plot(xmerge, np.ones(len(xmerge))*5., 'k*', label="BSM") 


plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='major',width = 1, length=5)
plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='minor',width = 0.75, length=2.5)
#plt.locator_params(axis='x', nbins=6)
plt.axes().xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.axes().yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.xticks(fontsize=ls)
plt.yticks(fontsize=ls) 
#plt.ylabel('Time (s)')
plt.xlabel('(dd/mmTHH:MM:SS)')
time_formatter = matplotlib.dates.DateFormatter("%d/%mT%H:%M:%S")
plt.axes().xaxis.set_major_formatter(time_formatter)
plt.axes().xaxis_date() # this is not actually necessary
plt.locator_params(axis='x', nbins=4)

plt.axes().set_yticks([1,2,3,4,5])
plt.axes().set_yticklabels(['BS1','BS2','BS3','BS4', 'BSM'], minor=False)

#plt.legend()

#plt.xlabel('Detection number',size=ls)
#plt.ylabel('Timestamp (s)',size=ls)
#plt.axis('equal')
plt.show()
    