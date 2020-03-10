# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 08:12:17 2020

Script to correct the clock drift presented on Vemco receivers during a deployment 

@author: Ivan Masmitja
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import calendar
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib
from scipy.optimize import curve_fit
import os
'''
#####################################################################################################
#####################################################################################################
#####################################################################################################
Here we specify the main parameters to synchronize the Vemco receivers. Each mooring must have a Vemco receiver 
and a syncTAG associated. Moreover, the campaing should have a reference tag in the center of the configuration (C)

With that on mind, the parameters to modify in this file are:
    - STAG_X = Is the name of each sync tag associated at each Vemco receiver.
    - A1_INITIAL = Is the true position of the base stations array. This position will be used to compute the others.
    - AX_INITIAL = Are the positions initialy gest. These positions are only used to plot a figure, therefore, are not impresindible.
    - SAVE_FOLDER = The folder where the synchronized files will be saved. Each tag will be saved in one .txt file for each base station.
                    Moreover, a merge file for each tag will be generated where a unique tag detection at each time will be saved, independently
                    of the base station which was detected first.
    -filename_BSX = The path and name of each .csv file generatd by the Vemco receivers
    -R_ANGLE = Angle of base station array respect A1 in degrees
    -DEPTH_AX = Depth of base stations respect to A1 in meters
'''

#Main parametres of the deployment, which are used to synchronize the timestamps
STAG_A = 'A69-1601-65015' #A
STAG_B = 'A69-1601-65014' #B
STAG_C = 'A69-1602-65266' #C
STAG_D = 'A69-1601-60592' #D
STAG_E = 'A69-1601-60593' #E

A1_INITIAL = np.array([543957.71,4651491.78,30.]) #from USBL' ROV
A2_INITIAL = np.array([544094.56,4651484.2,25.]) #from USBL' ROV
A3_INITIAL = np.array([543955.9,4651359.9,20.]) #from USBL' ROV
A4_INITIAL = np.array([544110.4,4651336.7,20.]) #from USBL' ROV

R_ANGLE = -4  #-4 #in degrees

DEPTH_A2 = -5.
DEPTH_A3 = -10
DEPTH_A4 = -10

SOUND_VELOCITY = 1512. #from CTD

SAVE_FOLDER = 'TagDetectionsRESNEPtest'

#real vemco files. Data only when all the base stations were in the water
filename_BS1 = 'VemcoFiles\VR2W_134725_20191105_2.csv'
filename_BS2 = 'VemcoFiles\VR2W_134724_20191105_2.csv'
filename_BS3 = 'VemcoFiles\VR2AR_548128_20191108_2.csv'
filename_BS4 = 'VemcoFiles\VR2AR_548129_20191108_2.csv'

'''
#####################################################################################################
#####################################################################################################
#####################################################################################################
'''
#%%
######################################################################################################
# Functions
######################################################################################################

'''
Function to read the Vemco files
'''
def read_BS_file(filename):
    #open files
    try:    
        timestamp_BS, tagID_BS = np.loadtxt(filename,skiprows=1, delimiter=',',usecols=(0,2), unpack=True)
    except ValueError:
        tagID_BS = np.loadtxt(filename,skiprows=1, delimiter=',',usecols=(2), unpack=True,dtype=str)
        timestamp = np.loadtxt(filename,skiprows=1, delimiter=',',usecols=(0), unpack=True,dtype=str)
        timestamp_BS_s = []
        timestamp_BS_ms = []
        for t in timestamp:
            timestamp_BS_s.append(calendar.timegm(time.strptime(t[:-4], '%Y-%m-%d %H:%M:%S')))
            timestamp_BS_ms.append(float(t[-4:]))
        timestamp_BS = np.array(timestamp_BS_s) + np.array(timestamp_BS_ms) 
    #Searching for the different tag id received by the base station
    tag_names = []
    for tag in tagID_BS:
        if tag not in tag_names:
            tag_names.append(tag)
    print 'The following tag ID have been found in this receiver file:', filename
    print tag_names
    #Create a list structure were each tag id has its own time stamp column
    detections_BS = []
    for tag in tag_names:
        aux = []
        aux.append(tag)
        for i in range(tagID_BS.size):
            if tagID_BS[i] == tag:
                aux.append(timestamp_BS[i])
        detections_BS.append(aux)
    return detections_BS,tag_names


'''
functions to adjust the clock drift using linear or polynomial line fitting
'''
def clock_drifft_adjust(x, pop):
    aux = np.ones(x.size)
    a = pop.item(0)
    b = pop.item(1)
    c = pop.item(2)
    d = pop.item(3)
    for i in range(x.size):
        aux[i] = a*x[i]**3 + b*x[i]**2 +c*x[i] + d
    return  aux

def clock_drifft_adjust_s(x, pop):
    aux = np.zeros(x.size)
    for i in range(x.size):
        for n in range(pop.size-1):
            aux[i] += pop.item(n)*x[i]**(pop.size-n-1)
        aux[i] += pop.item(n+1)
    return  aux


"""
create a function to fit with your data. a, b, c and d are the coefficients
that curve_fit will calculate for you. 
In this part you need to guess and/or use mathematical knowledge to find
a function that resembles your data
"""
def func(x, a, b, c, d):
    return a*x**3 + b*x**2 +c*x + d

def func_2(x, a, b):
    return a*x + b


'''
Function to find the timestamp of specific tag for each base station
'''
def find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4):
    #for each tag, see if it has been received. If so, save the timestamp array
    BS1 = False
    BS2 = False
    BS3 = False
    BS4 = False
    for i in range(len(tag_names_BS1)):
        if detections_BS1[i][0] == tag:
            timestamps_BS1 = np.array(detections_BS1[i][1:])
            BS1 = True
    for i in range(len(tag_names_BS2)):
        if detections_BS2[i][0] == tag:
            timestamps_BS2 = np.array(detections_BS2[i][1:])
            BS2 = True
    for i in range(len(tag_names_BS3)):
        if detections_BS3[i][0] == tag:
            timestamps_BS3 = np.array(detections_BS3[i][1:])
            BS3 = True
    for i in range(len(tag_names_BS4)):
        if detections_BS4[i][0] == tag:
            timestamps_BS4 = np.array(detections_BS4[i][1:])
            BS4 = True
    
    if BS1 == True and BS2 == True and BS3 == True and BS4 == True:
        warning = False
    else:
        warning = True
    return (warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4)

'''
Function to find the Time Difference Of Arrival (TDOA) of a tag at each Base station
'''
def find_tdoa_tagBS(timestamps_BS1_d,timestamps_BS2_d,timestamps_BS3_d,timestamps_BS4_d):
    #Computing the time difference between receptions
    a = 0
    b = 0
    c = 0
    d = 0
    threshold = 15
    timestamps_BS1_original = []
    timestamps_BS1_f = []
    timestamps_BS2_f = []
    timestamps_BS3_f = []
    timestamps_BS4_f = []
    while (True):
        t_min = 1e12
        try:
            if timestamps_BS1_d[a] < t_min:
                t_min = timestamps_BS1_d[a]
        except:
            next
        try:
            if timestamps_BS2_d[b] < t_min:
                t_min = timestamps_BS2_d[b]
        except:
            next
        try:
            if timestamps_BS3_d[c] < t_min:
                t_min = timestamps_BS3_d[c]
        except:
            next
        try:
            if timestamps_BS4_d[d] < t_min:
                t_min = timestamps_BS4_d[d]
        except:
            next
        if t_min == 1e12:
            break
        rt_aux = []
        BS_aux = []
        BS1 = False
        BS2 = False
        BS3 = False
        BS4 = False
        try:
            if timestamps_BS1_d[a] >= t_min-threshold and timestamps_BS1_d[a] <= t_min+threshold:
                rt_aux.append(timestamps_BS1_d[a])
                BS1 = True
                BS_aux.append(A1)
                a += 1
        except:
            next
        try:
            if timestamps_BS2_d[b] >= t_min-threshold and timestamps_BS2_d[b] <= t_min+threshold:
                rt_aux.append(timestamps_BS2_d[b])
                BS2 = True
                BS_aux.append(A2)
                b += 1
    #                print'BS2=Detection',b
        except:
            next
        try:
            if timestamps_BS3_d[c] >= t_min-threshold and timestamps_BS3_d[c] <= t_min+threshold:
                rt_aux.append(timestamps_BS3_d[c])
                BS3 = True
                BS_aux.append(A3)
                c += 1
    #                print'BS3=Detection',c
        except:
            next
        try:
            if timestamps_BS4_d[d] >= t_min-threshold and timestamps_BS4_d[d] <= t_min+threshold:
                rt_aux.append(timestamps_BS4_d[d])
                BS4 = True
                BS_aux.append(A4)
                d += 1
    #                print'BS4=Detection',d
        except:
            next
        #append the timestamps if the tag has been detected by the 4 receivers
        if BS1 == True and BS2 == True and BS3 == True and BS4 == True:
            timestamps_BS1_original.append(timestamps_BS1_d[a-1])
            timestamps_BS1_f.append(timestamps_BS1_d[a-1])
            timestamps_BS2_f.append(timestamps_BS2_d[b-1])
            timestamps_BS3_f.append(timestamps_BS3_d[c-1])
            timestamps_BS4_f.append(timestamps_BS4_d[d-1])
            
    y12 = np.array(timestamps_BS1_f)-np.array(timestamps_BS2_f)
    y13 = np.array(timestamps_BS1_f)-np.array(timestamps_BS3_f)
    y14 = np.array(timestamps_BS1_f)-np.array(timestamps_BS4_f)
    y23 = np.array(timestamps_BS2_f)-np.array(timestamps_BS3_f)
    y24 = np.array(timestamps_BS2_f)-np.array(timestamps_BS4_f)
    y34 = np.array(timestamps_BS3_f)-np.array(timestamps_BS4_f)
    return (y12,y13,y14,y23,y24,y34,timestamps_BS1_original)

#%%
#####################################################################################################
#    MAIN PROGRAM
#####################################################################################################

#%%
#Read Vemco files
print 'Reading the Vemco files'
detections_BS1,tag_names_BS1 = read_BS_file(filename_BS1)
detections_BS2,tag_names_BS2 = read_BS_file(filename_BS2)
detections_BS3,tag_names_BS3 = read_BS_file(filename_BS3)
detections_BS4,tag_names_BS4 = read_BS_file(filename_BS4)


#%%   
# first step to adjust the clock drift   
print 'Start clock syncHronization step 0: Linear regresion'
tag = STAG_C
warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4 = find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4)
if warning == True:
    print 'WARNING: Some BaseStation has not detected the tag'

#to maintain the nomenclature
timestamps_BS1_i = timestamps_BS1 + 0.
timestamps_BS2_i = timestamps_BS2 + 0.
timestamps_BS3_i = timestamps_BS3 + 0.
timestamps_BS4_i = timestamps_BS4 + 0.

#Computing the time difference between receptions
y12,y13,y14,y23,y24,y34,timestamps_BS1_original = find_tdoa_tagBS(timestamps_BS1_i,timestamps_BS2_i,timestamps_BS3_i,timestamps_BS4_i)

#make the curve_fit
x1 = np.array(timestamps_BS1_original)
popt12, pcov12 = curve_fit(func_2, x1, y12)
popt13, pcov13 = curve_fit(func_2, x1, y13)
popt14, pcov14 = curve_fit(func_2, x1, y14)

#plot results
x = [datetime.utcfromtimestamp(element) for element in x1]
ls=12
fig = plt.figure(figsize=(5,5))
plt.plot(x,y12,'ro',ms=5,lw=1,label='C12') 
plt.plot(x,y13,'b^',ms=5,lw=1,label='C13') 
plt.plot(x,y14,'gs',ms=5,lw=1,label='C14') 
plt.plot(x, func_2(x1, *popt12)) #same as line above \/
plt.plot(x, func_2(x1, *popt13)) #same as line above \/
plt.plot(x, func_2(x1, *popt14)) #same as line above \/
plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='major',width = 1, length=5)
plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='minor',width = 0.75, length=2.5)
plt.locator_params(axis='x', nbins=6)
plt.axes().xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.axes().yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.xticks(fontsize=ls)
plt.yticks(fontsize=ls) 
plt.title('Clock drift 0')
plt.ylabel('Time (s)')
plt.xlabel('Date (dd/mm)')
plt.locator_params(axis='x', nbins=2)
time_formatter = matplotlib.dates.DateFormatter("%d/%m")
plt.axes().xaxis.set_major_formatter(time_formatter)
plt.axes().xaxis_date() # this is not actually necessary
plt.legend()
plt.savefig('clock_drift_0.jpg',format='jpg', dpi=200 ,bbox_inches='tight')
plt.show()    

##############################
#Clock drift parameters obtained from curve fitting
SLOPE  = np.array([popt12.item(0), popt13.item(0), popt14.item(0)])
OFFSET = np.array([popt12.item(1), popt13.item(1), popt14.item(1)])
##############################
#%%

print 'Start clock synchronization step 1: Polinomial regresion'
tag = STAG_C
warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4 = find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4)
if warning == True:
    print 'WARNING: Some BaseStation has not detected the tag'
        
#To maintain nomenclature
timestamps_BS1_i = timestamps_BS1 + 0.
timestamps_BS2_i = timestamps_BS2 + 0.
timestamps_BS3_i = timestamps_BS3 + 0.
timestamps_BS4_i = timestamps_BS4 + 0.

#Correct the clocks drifft (SLOPE,OFFSET): corese adjust using the previous computed parameters with liner curve fitting
timestamps_BS1_d = timestamps_BS1_i + 0.
timestamps_BS2_d = timestamps_BS2_i + (timestamps_BS2_i*SLOPE[0]+OFFSET[0])
timestamps_BS3_d = timestamps_BS3_i + (timestamps_BS3_i*SLOPE[1]+OFFSET[1])
timestamps_BS4_d = timestamps_BS4_i + (timestamps_BS4_i*SLOPE[2]+OFFSET[2])
 
#Computing the time difference between receptions
y12,y13,y14,y23,y24,y34,timestamps_BS1_original = find_tdoa_tagBS(timestamps_BS1_d,timestamps_BS2_d,timestamps_BS3_d,timestamps_BS4_d)
            
#make the curve_fit (polinomial)
x1 = np.array(timestamps_BS1_original)
try:
    popt12, pcov12 = curve_fit(func, x1, y12)
except:
    # if y1X has some positive and negative values, curve_fit can not find an optimal solution
    popt12, pcov12 = curve_fit(func, x1, abs(y12))
try:
    popt13, pcov13 = curve_fit(func, x1, y13)
except:
    # if y1X has some positive and negative values, curve_fit can not find an optimal solution
    popt13, pcov13 = curve_fit(func, x1, abs(y13))
try:
    popt14, pcov14 = curve_fit(func, x1, y14)
except:
    # if y1X has some positive and negative values, curve_fit can not find an optimal solution
    popt14, pcov14 = curve_fit(func, x1, abs(y14))

#plot data
x = [datetime.utcfromtimestamp(element) for element in x1]
ls=12
fig = plt.figure(figsize=(5,5))
plt.plot(x,y12,'ro',ms=5,lw=1) 
plt.plot(x,y13,'bo',ms=5,lw=1) 
plt.plot(x,y14,'go',ms=5,lw=1) 
plt.plot(x, func(x1, *popt12), label="Fitted Curve") #same as line above \/
plt.plot(x, func(x1, *popt13), label="Fitted Curve") #same as line above \/
plt.plot(x, func(x1, *popt14), label="Fitted Curve") #same as line above \/
plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='major',width = 1, length=5)
plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='minor',width = 0.75, length=2.5)
plt.locator_params(axis='x', nbins=6)
plt.axes().xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.axes().yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.xticks(fontsize=ls)
plt.yticks(fontsize=ls) 
plt.title('Clock drift 1')
plt.ylabel('Time (s)')
plt.xlabel('Date (dd/mm)')
plt.locator_params(axis='x', nbins=2)
time_formatter = matplotlib.dates.DateFormatter("%d/%m")
plt.axes().xaxis.set_major_formatter(time_formatter)
plt.axes().xaxis_date() # this is not actually necessary
plt.show()    

##############################################
#Save the values computed
pop_12 = popt12 + 0.
pop_13 = popt13 + 0.
pop_14 = popt14 + 0.
##############################################
#%%

print 'Start clock synchronization step 2: Polinomial regresion with windowing'
tag = STAG_C
warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4 = find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4)
if warning == True:
    print 'WARNING: Some BaseStation has not detected the tag'
        
#To maintain nomenclature
timestamps_BS1_i = timestamps_BS1 + 0.
timestamps_BS2_i = timestamps_BS2 + 0.
timestamps_BS3_i = timestamps_BS3 + 0.
timestamps_BS4_i = timestamps_BS4 + 0.

#Correct the clocks drifft (SLOPE,OFFSET): corese adjust using linear curve fitting
timestamps_BS1_d0 = timestamps_BS1_i + 0.
timestamps_BS2_d0 = timestamps_BS2_i + (timestamps_BS2_i*SLOPE[0]+OFFSET[0])
timestamps_BS3_d0 = timestamps_BS3_i + (timestamps_BS3_i*SLOPE[1]+OFFSET[1])
timestamps_BS4_d0 = timestamps_BS4_i + (timestamps_BS4_i*SLOPE[2]+OFFSET[2])

#Correct the clocks drifft step 1: fine adjust using curve fitting
timestamps_BS1_d1 = timestamps_BS1_d0 + 0.
timestamps_BS2_d1 = timestamps_BS2_d0 + clock_drifft_adjust(timestamps_BS2_d0,pop_12)
timestamps_BS3_d1 = timestamps_BS3_d0 + clock_drifft_adjust(timestamps_BS3_d0,pop_13)
timestamps_BS4_d1 = timestamps_BS4_d0 + clock_drifft_adjust(timestamps_BS4_d0,pop_14)  
 
#Computing the time difference between receptions
y12,y13,y14,y23,y24,y34,timestamps_BS1_original = find_tdoa_tagBS(timestamps_BS1_d1,timestamps_BS2_d1,timestamps_BS3_d1,timestamps_BS4_d1)
            
#make the curve_fit (using windowing)
x1 = np.array(timestamps_BS1_original)
num_splits = 7
x1_s = np.split(x1,num_splits)
y12_s = np.split(y12,num_splits)
y13_s = np.split(y13,num_splits)
y14_s = np.split(y14,num_splits)
popt12_s = []
popt13_s = []
popt14_s = []
for i in range(num_splits):
    popt12_s.append(np.poly1d(np.polyfit(x1_s[i], y12_s[i], 5)))
    popt13_s.append(np.poly1d(np.polyfit(x1_s[i], y13_s[i], 5)))
    popt14_s.append(np.poly1d(np.polyfit(x1_s[i], y14_s[i], 5)))
    
#plot data
x = [datetime.utcfromtimestamp(element) for element in x1]
ls=12
fig = plt.figure(figsize=(5,5))
plt.plot(x,y12,'ro',ms=5,lw=1) 
plt.plot(x,y13,'bo',ms=5,lw=1) 
plt.plot(x,y14,'go',ms=5,lw=1) 
for i in range(num_splits):
    plt.plot(np.split(np.array(x),num_splits)[i], popt12_s[i](x1_s[i]), label="Fitted Curve") 
    plt.plot(np.split(np.array(x),num_splits)[i], popt13_s[i](x1_s[i]), label="Fitted Curve")
    plt.plot(np.split(np.array(x),num_splits)[i], popt14_s[i](x1_s[i]), label="Fitted Curve") 
plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='major',width = 1, length=5)
plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='minor',width = 0.75, length=2.5)
plt.locator_params(axis='x', nbins=6)
plt.axes().xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.axes().yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.xticks(fontsize=ls)
plt.yticks(fontsize=ls) 
plt.title('Clock drift 1')
plt.ylabel('Time (s)')
plt.xlabel('Date (dd/mm)')
plt.locator_params(axis='x', nbins=2)
time_formatter = matplotlib.dates.DateFormatter("%d/%m")
plt.axes().xaxis.set_major_formatter(time_formatter)
plt.axes().xaxis_date() # this is not actually necessary
plt.show()    

##################################################
#Save the parameters to correct the clocks drifft: ultra-fine adjust using Python cureve fitting
pop_12_s = popt12_s
pop_13_s = popt13_s
pop_14_s = popt14_s
##################################################

#Correct the clocks drifft: ultrafine adjust using polinomial curve fitting by windowing
timestamps_BS1_d2 = timestamps_BS1_d1 + 0.
timestamps_BS2_d2 = np.array([])
timestamps_BS3_d2 = np.array([])
timestamps_BS4_d2 = np.array([])
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS2_d1[high] >= x1_s[i][-1] or high == timestamps_BS2_d1.size-1:
            break
        high += 1
    aux_BS2 = timestamps_BS2_d1[low:high] + clock_drifft_adjust_s(timestamps_BS2_d1[low:high],np.array(pop_12_s[i]))
    timestamps_BS2_d2 = np.concatenate((timestamps_BS2_d2,aux_BS2),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS3_d1[high] >= x1_s[i][-1] or high == timestamps_BS3_d1.size-1:
            break
        high += 1
    aux_BS3 = timestamps_BS3_d1[low:high] + clock_drifft_adjust_s(timestamps_BS3_d1[low:high],np.array(pop_13_s[i]))
    timestamps_BS3_d2 = np.concatenate((timestamps_BS3_d2,aux_BS3),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS4_d1[high] >= x1_s[i][-1] or high == timestamps_BS4_d1.size-1:
            break
        high += 1
    aux_BS4 = timestamps_BS4_d1[low:high] + clock_drifft_adjust_s(timestamps_BS4_d1[low:high],np.array(pop_14_s[i]))
    timestamps_BS4_d2 = np.concatenate((timestamps_BS4_d2,aux_BS4),axis=0)
    low = high + 0

#Computing the time difference between receptions
y12,y13,y14,y23,y24,y34,timestamps_BS1_original = find_tdoa_tagBS(timestamps_BS1_d2,timestamps_BS2_d2,timestamps_BS3_d2,timestamps_BS4_d2)
x1 = np.array(timestamps_BS1_original)
#plot data
x = [datetime.utcfromtimestamp(element) for element in x1]
ls=12
fig = plt.figure(figsize=(5,5))
plt.plot(x,y12,'ro--',ms=5,lw=1) 
plt.plot(x,y13,'bo--',ms=5,lw=1) 
plt.plot(x,y14,'go--',ms=5,lw=1) 
plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='major',width = 1, length=5)
plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='minor',width = 0.75, length=2.5)
plt.locator_params(axis='x', nbins=6)
plt.axes().xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.axes().yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.xticks(fontsize=ls)
plt.yticks(fontsize=ls) 
plt.title('Clock drift 1')
plt.ylabel('Time (s)')
plt.xlabel('Date (dd/mm HH:MM)')
plt.locator_params(axis='x', nbins=2)
time_formatter = matplotlib.dates.DateFormatter("%d/%m %H:%M")
plt.axes().xaxis.set_major_formatter(time_formatter)
plt.axes().xaxis_date() # this is not actually necessary
plt.show()    


#%%
print 'Start clock synchronization step 3: Equalize time of flight between pairs of synctag and base station'
tag = STAG_A
warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4 = find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4)
if warning == True:
    print 'WARNING: Some BaseStation has not detected the tag'

#To maintain nomenclature
timestamps_BS1_i = timestamps_BS1 + 0.
timestamps_BS2_i = timestamps_BS2 + 0.
timestamps_BS3_i = timestamps_BS3 + 0.
timestamps_BS4_i = timestamps_BS4 + 0.

#Correct the clocks drifft (SLOPE,OFFSET): corese adjust using liner curve fitting
timestamps_BS1_d0 = timestamps_BS1_i + 0.
timestamps_BS2_d0 = timestamps_BS2_i + (timestamps_BS2_i*SLOPE[0]+OFFSET[0])
timestamps_BS3_d0 = timestamps_BS3_i + (timestamps_BS3_i*SLOPE[1]+OFFSET[1])
timestamps_BS4_d0 = timestamps_BS4_i + (timestamps_BS4_i*SLOPE[2]+OFFSET[2])

#Correct the clocks drifft: fine adjust using polinomial curve fitting
timestamps_BS1_d1 = timestamps_BS1_d0 + 0.
timestamps_BS2_d1 = timestamps_BS2_d0 + clock_drifft_adjust(timestamps_BS2_d0,pop_12)
timestamps_BS3_d1 = timestamps_BS3_d0 + clock_drifft_adjust(timestamps_BS3_d0,pop_13)
timestamps_BS4_d1 = timestamps_BS4_d0 + clock_drifft_adjust(timestamps_BS4_d0,pop_14)   

#Correct the clocks drifft: ultra-fine adjust using polinomial curve fitting by windowing
timestamps_BS1_d = timestamps_BS1_d1 + 0.
timestamps_BS2_d = np.array([])
timestamps_BS3_d = np.array([])
timestamps_BS4_d = np.array([])
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS2_d1[high] >= x1_s[i][-1] or high == timestamps_BS2_d1.size-1:
            break
        high += 1
    aux_BS2 = timestamps_BS2_d1[low:high] + clock_drifft_adjust_s(timestamps_BS2_d1[low:high],np.array(pop_12_s[i]))
    timestamps_BS2_d = np.concatenate((timestamps_BS2_d,aux_BS2),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS3_d1[high] >= x1_s[i][-1] or high == timestamps_BS3_d1.size-1:
            break
        high += 1
    aux_BS3 = timestamps_BS3_d1[low:high] + clock_drifft_adjust_s(timestamps_BS3_d1[low:high],np.array(pop_13_s[i]))
    timestamps_BS3_d = np.concatenate((timestamps_BS3_d,aux_BS3),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS4_d1[high] >= x1_s[i][-1] or high == timestamps_BS4_d1.size-1:
            break
        high += 1
    aux_BS4 = timestamps_BS4_d1[low:high] + clock_drifft_adjust_s(timestamps_BS4_d1[low:high],np.array(pop_14_s[i]))
    timestamps_BS4_d = np.concatenate((timestamps_BS4_d,aux_BS4),axis=0)
    low = high + 0                      

#Computing the time difference between receptions
y12,y13,y14,y23,y24,y34,timestamps_BS1_original = find_tdoa_tagBS(timestamps_BS1_d,timestamps_BS2_d,timestamps_BS3_d,timestamps_BS4_d)

#Measure TOF mean between transponders
t_12 = np.mean(y12)
t_13 = np.mean(y13)
t_14 = np.mean(y14)

tag = STAG_B
warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4 = find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4)
if warning == True:
    print 'WARNING: Some BaseStation has not detected the tag'

#To maintain nomenclature
timestamps_BS1_i = timestamps_BS1 + 0.
timestamps_BS2_i = timestamps_BS2 + 0.
timestamps_BS3_i = timestamps_BS3 + 0.
timestamps_BS4_i = timestamps_BS4 + 0.

#Correct the clocks drifft (SLOPE,OFFSET): corese adjust using liner curve fitting
timestamps_BS1_d0 = timestamps_BS1_i + 0.
timestamps_BS2_d0 = timestamps_BS2_i + (timestamps_BS2_i*SLOPE[0]+OFFSET[0])
timestamps_BS3_d0 = timestamps_BS3_i + (timestamps_BS3_i*SLOPE[1]+OFFSET[1])
timestamps_BS4_d0 = timestamps_BS4_i + (timestamps_BS4_i*SLOPE[2]+OFFSET[2])

#Correct the clocks drifft: fine adjust using polinomial curve fitting
timestamps_BS1_d1 = timestamps_BS1_d0 + 0.
timestamps_BS2_d1 = timestamps_BS2_d0 + clock_drifft_adjust(timestamps_BS2_d0,pop_12)
timestamps_BS3_d1 = timestamps_BS3_d0 + clock_drifft_adjust(timestamps_BS3_d0,pop_13)
timestamps_BS4_d1 = timestamps_BS4_d0 + clock_drifft_adjust(timestamps_BS4_d0,pop_14)

#Correct the clocks drifft: ultrafine adjust using polinomial curve fitting by windowing
timestamps_BS1_d = timestamps_BS1_d1 + 0.
timestamps_BS2_d = np.array([])
timestamps_BS3_d = np.array([])
timestamps_BS4_d = np.array([])
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS2_d1[high] >= x1_s[i][-1] or high == timestamps_BS2_d1.size-1:
            break
        high += 1
    aux_BS2 = timestamps_BS2_d1[low:high] + clock_drifft_adjust_s(timestamps_BS2_d1[low:high],np.array(pop_12_s[i]))
    timestamps_BS2_d = np.concatenate((timestamps_BS2_d,aux_BS2),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS3_d1[high] >= x1_s[i][-1] or high == timestamps_BS3_d1.size-1:
            break
        high += 1
    aux_BS3 = timestamps_BS3_d1[low:high] + clock_drifft_adjust_s(timestamps_BS3_d1[low:high],np.array(pop_13_s[i]))
    timestamps_BS3_d = np.concatenate((timestamps_BS3_d,aux_BS3),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS4_d1[high] >= x1_s[i][-1] or high == timestamps_BS4_d1.size-1:
            break
        high += 1
    aux_BS4 = timestamps_BS4_d1[low:high] + clock_drifft_adjust_s(timestamps_BS4_d1[low:high],np.array(pop_14_s[i]))
    timestamps_BS4_d = np.concatenate((timestamps_BS4_d,aux_BS4),axis=0)
    low = high + 0   

#Computing the time difference between receptions
y12,y13,y14,y23,y24,y34,timestamps_BS1_original = find_tdoa_tagBS(timestamps_BS1_d,timestamps_BS2_d,timestamps_BS3_d,timestamps_BS4_d)

#Measure TOF mean between transponders
t_21 = np.mean(y12)

tag = STAG_D
warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4 = find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4)
if warning == True:
    print 'WARNING: Some BaseStation has not detected the tag'

#To maintain nomenclature
timestamps_BS1_i = timestamps_BS1 + 0.
timestamps_BS2_i = timestamps_BS2 + 0.
timestamps_BS3_i = timestamps_BS3 + 0.
timestamps_BS4_i = timestamps_BS4 + 0.

#Correct the clocks drifft (SLOPE,OFFSET): corese adjust using Excel liner curve fitting
timestamps_BS1_d0 = timestamps_BS1_i + 0.
timestamps_BS2_d0 = timestamps_BS2_i + (timestamps_BS2_i*SLOPE[0]+OFFSET[0])
timestamps_BS3_d0 = timestamps_BS3_i + (timestamps_BS3_i*SLOPE[1]+OFFSET[1])
timestamps_BS4_d0 = timestamps_BS4_i + (timestamps_BS4_i*SLOPE[2]+OFFSET[2])

#Correct the clocks drifft: fine adjust using polinomial curve fitting
timestamps_BS1_d1 = timestamps_BS1_d0 + 0.
timestamps_BS2_d1 = timestamps_BS2_d0 + clock_drifft_adjust(timestamps_BS2_d0,pop_12)
timestamps_BS3_d1 = timestamps_BS3_d0 + clock_drifft_adjust(timestamps_BS3_d0,pop_13)
timestamps_BS4_d1 = timestamps_BS4_d0 + clock_drifft_adjust(timestamps_BS4_d0,pop_14)

#Correct the clocks drifft: ultrafine adjust using polinomial curve fitting by windowing
timestamps_BS1_d = timestamps_BS1_d1 + 0.
timestamps_BS2_d = np.array([])
timestamps_BS3_d = np.array([])
timestamps_BS4_d = np.array([])
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS2_d1[high] >= x1_s[i][-1] or high == timestamps_BS2_d1.size-1:
            break
        high += 1
    aux_BS2 = timestamps_BS2_d1[low:high] + clock_drifft_adjust_s(timestamps_BS2_d1[low:high],np.array(pop_12_s[i]))
    timestamps_BS2_d = np.concatenate((timestamps_BS2_d,aux_BS2),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS3_d1[high] >= x1_s[i][-1] or high == timestamps_BS3_d1.size-1:
            break
        high += 1
    aux_BS3 = timestamps_BS3_d1[low:high] + clock_drifft_adjust_s(timestamps_BS3_d1[low:high],np.array(pop_13_s[i]))
    timestamps_BS3_d = np.concatenate((timestamps_BS3_d,aux_BS3),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS4_d1[high] >= x1_s[i][-1] or high == timestamps_BS4_d1.size-1:
            break
        high += 1
    aux_BS4 = timestamps_BS4_d1[low:high] + clock_drifft_adjust_s(timestamps_BS4_d1[low:high],np.array(pop_14_s[i]))
    timestamps_BS4_d = np.concatenate((timestamps_BS4_d,aux_BS4),axis=0)
    low = high + 0   

#Computing the time difference between receptions
y12,y13,y14,y23,y24,y34,timestamps_BS1_original = find_tdoa_tagBS(timestamps_BS1_d,timestamps_BS2_d,timestamps_BS3_d,timestamps_BS4_d)

#Measure TOF mean between transponders
t_31 = np.mean(y13)

tag = STAG_E
warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4 = find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4)
if warning == True:
    print 'WARNING: Some BaseStation has not detected the tag'

#To maintain nomenclature
timestamps_BS1_i = timestamps_BS1 + 0.
timestamps_BS2_i = timestamps_BS2 + 0.
timestamps_BS3_i = timestamps_BS3 + 0.
timestamps_BS4_i = timestamps_BS4 + 0.

#Correct the clocks drifft (SLOPE,OFFSET): corese adjust using Excel liner curve fitting
timestamps_BS1_d0 = timestamps_BS1_i + 0.
timestamps_BS2_d0 = timestamps_BS2_i + (timestamps_BS2_i*SLOPE[0]+OFFSET[0])
timestamps_BS3_d0 = timestamps_BS3_i + (timestamps_BS3_i*SLOPE[1]+OFFSET[1])
timestamps_BS4_d0 = timestamps_BS4_i + (timestamps_BS4_i*SLOPE[2]+OFFSET[2])

#Correct the clocks drifft: fine adjust using polinomial curve fitting
timestamps_BS1_d = timestamps_BS1_d0 + 0.
timestamps_BS2_d = timestamps_BS2_d0 + clock_drifft_adjust(timestamps_BS2_d0,pop_12)
timestamps_BS3_d = timestamps_BS3_d0 + clock_drifft_adjust(timestamps_BS3_d0,pop_13)
timestamps_BS4_d = timestamps_BS4_d0 + clock_drifft_adjust(timestamps_BS4_d0,pop_14)

#Computing the time difference between receptions
y12,y13,y14,y23,y24,y34,timestamps_BS1_original = find_tdoa_tagBS(timestamps_BS1_d,timestamps_BS2_d,timestamps_BS3_d,timestamps_BS4_d)

#Measure TOF mean between transponders
t_41 = np.mean(y14)

##############################################
#Finally, we save the values
dif_12_21 = (t_12+t_21)/2.
dif_13_31 = (t_13+t_31)/2.
dif_14_41 = (t_14+t_41)/2.
##############################################

#%%
print 'Start clock synchronization: Plot result'
tag = STAG_D
warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4 = find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4)
if warning == True:
    print 'WARNING: Some BaseStation has not detected the tag'

#To maintain nomenclature
timestamps_BS1_i = timestamps_BS1 + 0.
timestamps_BS2_i = timestamps_BS2 + 0.
timestamps_BS3_i = timestamps_BS3 + 0.
timestamps_BS4_i = timestamps_BS4 + 0.

#Correct the clocks drifft (SLOPE,OFFSET): corese adjust using liner curve fitting
timestamps_BS1_d0 = timestamps_BS1_i + 0.
timestamps_BS2_d0 = timestamps_BS2_i + (timestamps_BS2_i*SLOPE[0]+OFFSET[0])
timestamps_BS3_d0 = timestamps_BS3_i + (timestamps_BS3_i*SLOPE[1]+OFFSET[1])
timestamps_BS4_d0 = timestamps_BS4_i + (timestamps_BS4_i*SLOPE[2]+OFFSET[2])
        
#Correct the clocks drifft: fine adjust using Python cureve fitting    
timestamps_BS1_d1 = timestamps_BS1_d0 + 0.
timestamps_BS2_d1 = timestamps_BS2_d0 + clock_drifft_adjust(timestamps_BS2_d0,pop_12)
timestamps_BS3_d1 = timestamps_BS3_d0 + clock_drifft_adjust(timestamps_BS3_d0,pop_13)
timestamps_BS4_d1 = timestamps_BS4_d0 + clock_drifft_adjust(timestamps_BS4_d0,pop_14)

#Correct the clocks drifft: ultrafine adjust using polinomial curve fitting by windowing
timestamps_BS1_d2 = timestamps_BS1_d1 + 0.
timestamps_BS2_d2 = np.array([])
timestamps_BS3_d2 = np.array([])
timestamps_BS4_d2 = np.array([])
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS2_d1[high] >= x1_s[i][-1] or high == timestamps_BS2_d1.size-1:
            break
        high += 1
    aux_BS2 = timestamps_BS2_d1[low:high] + clock_drifft_adjust_s(timestamps_BS2_d1[low:high],np.array(pop_12_s[i]))
    timestamps_BS2_d2 = np.concatenate((timestamps_BS2_d2,aux_BS2),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS3_d1[high] >= x1_s[i][-1] or high == timestamps_BS3_d1.size-1:
            break
        high += 1
    aux_BS3 = timestamps_BS3_d1[low:high] + clock_drifft_adjust_s(timestamps_BS3_d1[low:high],np.array(pop_13_s[i]))
    timestamps_BS3_d2 = np.concatenate((timestamps_BS3_d2,aux_BS3),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS4_d1[high] >= x1_s[i][-1] or high == timestamps_BS4_d1.size-1:
            break
        high += 1
    aux_BS4 = timestamps_BS4_d1[low:high] + clock_drifft_adjust_s(timestamps_BS4_d1[low:high],np.array(pop_14_s[i]))
    timestamps_BS4_d2 = np.concatenate((timestamps_BS4_d2,aux_BS4),axis=0)
    low = high + 0   
    
#Correct the clocks based on the computed time differences bwetween two pairs of syncTAG/BaseStations
timestamps_BS1_d = timestamps_BS1_d2 + 0.
timestamps_BS2_d = timestamps_BS2_d2 + dif_12_21
timestamps_BS3_d = timestamps_BS3_d2 + dif_13_31
timestamps_BS4_d = timestamps_BS4_d2 + dif_14_41

#Plot results        
fig = plt.figure(figsize=(5,5))
plt.plot(timestamps_BS1_i,np.ones(timestamps_BS1_i.size)*1,'ro', label = 'Clk_initial')
plt.plot(timestamps_BS2_i,np.ones(timestamps_BS2_i.size)*2,'ro')
plt.plot(timestamps_BS3_i,np.ones(timestamps_BS3_i.size)*3,'ro')
plt.plot(timestamps_BS4_i,np.ones(timestamps_BS4_i.size)*4,'ro')

plt.plot(timestamps_BS1_d0,np.ones(timestamps_BS1_d0.size)*1,'b^', label = 'Clk_drift0')
plt.plot(timestamps_BS2_d0,np.ones(timestamps_BS2_d0.size)*2,'b^')
plt.plot(timestamps_BS3_d0,np.ones(timestamps_BS3_d0.size)*3,'b^')
plt.plot(timestamps_BS4_d0,np.ones(timestamps_BS4_d0.size)*4,'b^')

plt.plot(timestamps_BS1_d1,np.ones(timestamps_BS1_d1.size)*1,'ms', label = 'Clk_drift1')
plt.plot(timestamps_BS2_d1,np.ones(timestamps_BS2_d1.size)*2,'ms')
plt.plot(timestamps_BS3_d1,np.ones(timestamps_BS3_d1.size)*3,'ms')
plt.plot(timestamps_BS4_d1,np.ones(timestamps_BS4_d1.size)*4,'ms')

plt.plot(timestamps_BS1_d2,np.ones(timestamps_BS1_d2.size)*1,'c*', label = 'Clk_drift2')
plt.plot(timestamps_BS2_d2,np.ones(timestamps_BS2_d2.size)*2,'c*')
plt.plot(timestamps_BS3_d2,np.ones(timestamps_BS3_d2.size)*3,'c*')
plt.plot(timestamps_BS4_d2,np.ones(timestamps_BS4_d2.size)*4,'c*')

plt.plot(timestamps_BS1_d,np.ones(timestamps_BS1_d.size)*1,'g<', label = 'Clk_drift3')
plt.plot(timestamps_BS2_d,np.ones(timestamps_BS2_d.size)*2,'g<')
plt.plot(timestamps_BS3_d,np.ones(timestamps_BS3_d.size)*3,'g<')
plt.plot(timestamps_BS4_d,np.ones(timestamps_BS4_d.size)*4,'g<')

plt.title('Clock drift correction')
plt.xlabel('Time (s)')
plt.axes().set_yticks([1,2,3,4])
plt.axes().set_yticklabels(['BS1','BS2','BS3','BS4'], minor=False)
plt.legend()
plt.show

#%%
print 'Compute the Base Station positions based on TOF'
tag = STAG_A
warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4 = find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4)
if warning == True:
    print 'WARNING: Some BaseStation has not detected the tag'

#To maintain nomenclature
timestamps_BS1_i = timestamps_BS1 + 0.
timestamps_BS2_i = timestamps_BS2 + 0.
timestamps_BS3_i = timestamps_BS3 + 0.
timestamps_BS4_i = timestamps_BS4 + 0.

#Correct the clocks drifft (SLOPE,OFFSET): corese adjust using liner curve fitting
timestamps_BS1_d0 = timestamps_BS1_i + 0.
timestamps_BS2_d0 = timestamps_BS2_i + (timestamps_BS2_i*SLOPE[0]+OFFSET[0])
timestamps_BS3_d0 = timestamps_BS3_i + (timestamps_BS3_i*SLOPE[1]+OFFSET[1])
timestamps_BS4_d0 = timestamps_BS4_i + (timestamps_BS4_i*SLOPE[2]+OFFSET[2])
        
#Correct the clocks drifft: fine adjust using Python cureve fitting    
timestamps_BS1_d1 = timestamps_BS1_d0 + 0.
timestamps_BS2_d1 = timestamps_BS2_d0 + clock_drifft_adjust(timestamps_BS2_d0,pop_12)
timestamps_BS3_d1 = timestamps_BS3_d0 + clock_drifft_adjust(timestamps_BS3_d0,pop_13)
timestamps_BS4_d1 = timestamps_BS4_d0 + clock_drifft_adjust(timestamps_BS4_d0,pop_14)

#Correct the clocks drifft: ultrafine adjust using polinomial curve fitting by windowing
timestamps_BS1_d2 = timestamps_BS1_d1 + 0.
timestamps_BS2_d2 = np.array([])
timestamps_BS3_d2 = np.array([])
timestamps_BS4_d2 = np.array([])
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS2_d1[high] >= x1_s[i][-1] or high == timestamps_BS2_d1.size-1:
            break
        high += 1
    aux_BS2 = timestamps_BS2_d1[low:high] + clock_drifft_adjust_s(timestamps_BS2_d1[low:high],np.array(pop_12_s[i]))
    timestamps_BS2_d2 = np.concatenate((timestamps_BS2_d2,aux_BS2),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS3_d1[high] >= x1_s[i][-1] or high == timestamps_BS3_d1.size-1:
            break
        high += 1
    aux_BS3 = timestamps_BS3_d1[low:high] + clock_drifft_adjust_s(timestamps_BS3_d1[low:high],np.array(pop_13_s[i]))
    timestamps_BS3_d2 = np.concatenate((timestamps_BS3_d2,aux_BS3),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS4_d1[high] >= x1_s[i][-1] or high == timestamps_BS4_d1.size-1:
            break
        high += 1
    aux_BS4 = timestamps_BS4_d1[low:high] + clock_drifft_adjust_s(timestamps_BS4_d1[low:high],np.array(pop_14_s[i]))
    timestamps_BS4_d2 = np.concatenate((timestamps_BS4_d2,aux_BS4),axis=0)
    low = high + 0   
    
#Correct the clocks based on the computed time differences bwetween two pairs of syncTAG/BaseStations
timestamps_BS1_d = timestamps_BS1_d2 + 0.
timestamps_BS2_d = timestamps_BS2_d2 + dif_12_21
timestamps_BS3_d = timestamps_BS3_d2 + dif_13_31
timestamps_BS4_d = timestamps_BS4_d2 + dif_14_41
#Computing the time difference between receptions
y12,y13,y14,y23,y24,y34,timestamps_BS1_original = find_tdoa_tagBS(timestamps_BS1_d,timestamps_BS2_d,timestamps_BS3_d,timestamps_BS4_d)

#Compute the new Vemco positions based on distances between pairs of BS. This has been computed using the
#time of flight of sync tags
d12 = y12*SOUND_VELOCITY
d13 = y13*SOUND_VELOCITY
d14 = y14*SOUND_VELOCITY

x1 = np.array(timestamps_BS1_original)
x_1 = [datetime.utcfromtimestamp(element) for element in x1]

tag = STAG_B
warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4 = find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4)
if warning == True:
    print 'WARNING: Some BaseStation has not detected the tag'

#To maintain nomenclature
timestamps_BS1_i = timestamps_BS1 + 0.
timestamps_BS2_i = timestamps_BS2 + 0.
timestamps_BS3_i = timestamps_BS3 + 0.
timestamps_BS4_i = timestamps_BS4 + 0.

#Correct the clocks drifft (SLOPE,OFFSET): corese adjust using liner curve fitting
timestamps_BS1_d0 = timestamps_BS1_i + 0.
timestamps_BS2_d0 = timestamps_BS2_i + (timestamps_BS2_i*SLOPE[0]+OFFSET[0])
timestamps_BS3_d0 = timestamps_BS3_i + (timestamps_BS3_i*SLOPE[1]+OFFSET[1])
timestamps_BS4_d0 = timestamps_BS4_i + (timestamps_BS4_i*SLOPE[2]+OFFSET[2])
       
#Correct the clocks drifft: fine adjust using Python cureve fitting    
timestamps_BS1_d1 = timestamps_BS1_d0 + 0.
timestamps_BS2_d1 = timestamps_BS2_d0 + clock_drifft_adjust(timestamps_BS2_d0,pop_12)
timestamps_BS3_d1 = timestamps_BS3_d0 + clock_drifft_adjust(timestamps_BS3_d0,pop_13)
timestamps_BS4_d1 = timestamps_BS4_d0 + clock_drifft_adjust(timestamps_BS4_d0,pop_14)

#Correct the clocks drifft: ultrafine adjust using polinomial curve fitting by windowing
timestamps_BS1_d2 = timestamps_BS1_d1 + 0.
timestamps_BS2_d2 = np.array([])
timestamps_BS3_d2 = np.array([])
timestamps_BS4_d2 = np.array([])
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS2_d1[high] >= x1_s[i][-1] or high == timestamps_BS2_d1.size-1:
            break
        high += 1
    aux_BS2 = timestamps_BS2_d1[low:high] + clock_drifft_adjust_s(timestamps_BS2_d1[low:high],np.array(pop_12_s[i]))
    timestamps_BS2_d2 = np.concatenate((timestamps_BS2_d2,aux_BS2),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS3_d1[high] >= x1_s[i][-1] or high == timestamps_BS3_d1.size-1:
            break
        high += 1
    aux_BS3 = timestamps_BS3_d1[low:high] + clock_drifft_adjust_s(timestamps_BS3_d1[low:high],np.array(pop_13_s[i]))
    timestamps_BS3_d2 = np.concatenate((timestamps_BS3_d2,aux_BS3),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS4_d1[high] >= x1_s[i][-1] or high == timestamps_BS4_d1.size-1:
            break
        high += 1
    aux_BS4 = timestamps_BS4_d1[low:high] + clock_drifft_adjust_s(timestamps_BS4_d1[low:high],np.array(pop_14_s[i]))
    timestamps_BS4_d2 = np.concatenate((timestamps_BS4_d2,aux_BS4),axis=0)
    low = high + 0   
    
#Correct the clocks based on the computed time differences bwetween two pairs of syncTAG/BaseStations
timestamps_BS1_d = timestamps_BS1_d2 + 0.
timestamps_BS2_d = timestamps_BS2_d2 + dif_12_21
timestamps_BS3_d = timestamps_BS3_d2 + dif_13_31
timestamps_BS4_d = timestamps_BS4_d2 + dif_14_41
#Computing the time difference between receptions
y12,y13,y14,y23,y24,y34,timestamps_BS1_original = find_tdoa_tagBS(timestamps_BS1_d,timestamps_BS2_d,timestamps_BS3_d,timestamps_BS4_d)

x1 = np.array(timestamps_BS1_original)
x_2 = [datetime.utcfromtimestamp(element) for element in x1]

#Compute the new Vemco positions based on distances between pairs of BS. This has been computed using the
#time of flight of sync tags
d23 = y23*SOUND_VELOCITY
d24 = y24*SOUND_VELOCITY

tag = STAG_E
warning, timestamps_BS1, timestamps_BS2, timestamps_BS3, timestamps_BS4 = find_timestamp_tagBS(tag,detections_BS1,detections_BS2,detections_BS3,detections_BS4,tag_names_BS1,tag_names_BS2,tag_names_BS3,tag_names_BS4)
if warning == True:
    print 'WARNING: Some BaseStation has not detected the tag'

#To maintain nomenclature
timestamps_BS1_i = timestamps_BS1 + 0.
timestamps_BS2_i = timestamps_BS2 + 0.
timestamps_BS3_i = timestamps_BS3 + 0.
timestamps_BS4_i = timestamps_BS4 + 0.

#Correct the clocks drifft (SLOPE,OFFSET): corese adjust using liner curve fitting
timestamps_BS1_d0 = timestamps_BS1_i + 0.
timestamps_BS2_d0 = timestamps_BS2_i + (timestamps_BS2_i*SLOPE[0]+OFFSET[0])
timestamps_BS3_d0 = timestamps_BS3_i + (timestamps_BS3_i*SLOPE[1]+OFFSET[1])
timestamps_BS4_d0 = timestamps_BS4_i + (timestamps_BS4_i*SLOPE[2]+OFFSET[2])
       
#Correct the clocks drifft: fine adjust using Python cureve fitting    
timestamps_BS1_d1 = timestamps_BS1_d0 + 0.
timestamps_BS2_d1 = timestamps_BS2_d0 + clock_drifft_adjust(timestamps_BS2_d0,pop_12)
timestamps_BS3_d1 = timestamps_BS3_d0 + clock_drifft_adjust(timestamps_BS3_d0,pop_13)
timestamps_BS4_d1 = timestamps_BS4_d0 + clock_drifft_adjust(timestamps_BS4_d0,pop_14)

#Correct the clocks drifft: ultrafine adjust using polinomial curve fitting by windowing
timestamps_BS1_d2 = timestamps_BS1_d1 + 0.
timestamps_BS2_d2 = np.array([])
timestamps_BS3_d2 = np.array([])
timestamps_BS4_d2 = np.array([])
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS2_d1[high] >= x1_s[i][-1] or high == timestamps_BS2_d1.size-1:
            break
        high += 1
    aux_BS2 = timestamps_BS2_d1[low:high] + clock_drifft_adjust_s(timestamps_BS2_d1[low:high],np.array(pop_12_s[i]))
    timestamps_BS2_d2 = np.concatenate((timestamps_BS2_d2,aux_BS2),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS3_d1[high] >= x1_s[i][-1] or high == timestamps_BS3_d1.size-1:
            break
        high += 1
    aux_BS3 = timestamps_BS3_d1[low:high] + clock_drifft_adjust_s(timestamps_BS3_d1[low:high],np.array(pop_13_s[i]))
    timestamps_BS3_d2 = np.concatenate((timestamps_BS3_d2,aux_BS3),axis=0)
    low = high + 0
low = 0
high = 0
for i in range(num_splits):
    # find the upper time limit to change the fitting parameters
    while (True):
        if timestamps_BS4_d1[high] >= x1_s[i][-1] or high == timestamps_BS4_d1.size-1:
            break
        high += 1
    aux_BS4 = timestamps_BS4_d1[low:high] + clock_drifft_adjust_s(timestamps_BS4_d1[low:high],np.array(pop_14_s[i]))
    timestamps_BS4_d2 = np.concatenate((timestamps_BS4_d2,aux_BS4),axis=0)
    low = high + 0   
    
#Correct the clocks based on the computed time differences bwetween two pairs of syncTAG/BaseStations
dif_12_21 = (t_12+t_21)/2.
dif_13_31 = (t_13+t_31)/2.
dif_14_41 = (t_14+t_41)/2.
timestamps_BS1_d = timestamps_BS1_d2 + 0.
timestamps_BS2_d = timestamps_BS2_d2 + dif_12_21
timestamps_BS3_d = timestamps_BS3_d2 + dif_13_31
timestamps_BS4_d = timestamps_BS4_d2 + dif_14_41
#Computing the time difference between receptions
y12,y13,y14,y23,y24,y34,timestamps_BS1_original = find_tdoa_tagBS(timestamps_BS1_d,timestamps_BS2_d,timestamps_BS3_d,timestamps_BS4_d)

x1 = np.array(timestamps_BS1_original)
x_3 = [datetime.utcfromtimestamp(element) for element in x1]

#Compute the new Vemco positions based on distances between pairs of BS. This has been computed using the
#time of flight of sync tags
d34 = y34*SOUND_VELOCITY

d_12 = abs(np.mean(d12))
d_13 = abs(np.mean(d13))
d_14 = abs(np.mean(d14))
d_23 = abs(np.mean(d23))
d_24 = abs(np.mean(d24))
d_34 = abs(np.mean(d34))

h_3 = np.sqrt(d_23**2-(d_12**2+d_23**2-d_13**2)**2/(4*d_12**2))
xd_3 = np.sqrt(d_13**2-h_3**2)

h_4 = np.sqrt(d_24**2-(d_12**2+d_24**2-d_14**2)**2/(4*d_12**2))
xd_4 = np.sqrt(d_14**2-h_4**2)

A1_sync = np.array([0,0,0.])
A2_sync = np.array([d_12,0.,DEPTH_A2])
A3_sync = np.array([xd_3,-h_3,DEPTH_A3])
A4_sync = np.array([xd_4,-h_4,DEPTH_A4])

#finally, here we adjust the initial position given by the USBL using the distances computed whith the TOF
#Translate the positions using trigonometry
A1 = A1_INITIAL + A1_sync
A2 = A1_INITIAL + A2_sync
A3 = A1_INITIAL + A3_sync
A4 = A1_INITIAL + A4_sync
#Rotate the positions given an angle
R_ANGLE = R_ANGLE*np.pi/180.
A2[0] = (A2[0]-A1[0])*np.cos(R_ANGLE)-(A2[1]-A1[1])*np.sin(R_ANGLE)+A1[0]
A2[1] = (A2[0]-A1[0])*np.sin(R_ANGLE)+(A2[1]-A1[1])*np.cos(R_ANGLE)+A1[1]
A3[0] = (A3[0]-A1[0])*np.cos(R_ANGLE)-(A3[1]-A1[1])*np.sin(R_ANGLE)+A1[0]
A3[1] = (A3[0]-A1[0])*np.sin(R_ANGLE)+(A3[1]-A1[1])*np.cos(R_ANGLE)+A1[1]
A4[0] = (A4[0]-A1[0])*np.cos(R_ANGLE)-(A4[1]-A1[1])*np.sin(R_ANGLE)+A1[0]
A4[1] = (A4[0]-A1[0])*np.sin(R_ANGLE)+(A4[1]-A1[1])*np.cos(R_ANGLE)+A1[1]

#save the base station computed positions
if os.path.exists(SAVE_FOLDER+'_BSP') == False:
    os.mkdir(SAVE_FOLDER+'_BSP')
aux_A = np.matrix(np.concatenate((A1,A2,A3,A4),axis=0)).reshape(4,3)
np.savetxt(SAVE_FOLDER+'_BSP/position_BS_.txt', aux_A, delimiter=',',header='Base station positions: Easting(m) Northing(m), Depth(m)')

print 'Distance between BS1-BS2 = %.3f (%.3f) m' % (d_12, np.std(d12))
print 'Distance between BS1-BS3 = %.3f (%.3f) m' % (d_13, np.std(d13))
print 'Distance between BS1-BS4 = %.3f (%.3f) m' % (d_14, np.std(d14))
print 'Distance between BS2-BS3 = %.3f (%.3f) m' % (d_23, np.std(d23))
print 'Distance between BS2-BS4 = %.3f (%.3f) m' % (d_24, np.std(d24))
print 'Distance between BS3-BS4 = %.3f (%.3f) m' % (d_34, np.std(d34))

#%%     
yes = True   
print 'Save a .txt file for each tag detected'

#Compute the time differences for each tag id at each base station
tag_names_total = list(set(tag_names_BS1) | set(tag_names_BS2) | set(tag_names_BS3) | set(tag_names_BS4))
timestamps_BS1 = np.array([])
timestamps_BS2 = np.array([])
timestamps_BS3 = np.array([])
timestamps_BS4 = np.array([])
timestamps_BS1_original = []
timestamps_BS1_f = []
timestamps_BS2_f = []
timestamps_BS3_f = []
timestamps_BS4_f = []
tag_tracked = []
for tag in tag_names_total:
#    print tag
#    if tag == 'A69-1601-65015': #A
#    if tag == 'A69-1601-65014': #B
#    if tag == 'A69-1602-65266': #C
#    if tag == 'A69-1601-60592': #D
#    if tag == 'A69-1601-60593': #E
#    if tag == 'A69-1602-14465': #escamarla
    if yes==True:
#        print tag
        #for each tag, see if it has been received. If so, save the timestamp array
        BS1 = False
        BS2 = False
        BS3 = False
        BS4 = False
        for i in range(len(tag_names_BS1)):
            if detections_BS1[i][0] == tag:
                timestamps_BS1 = np.array(detections_BS1[i][1:])
                BS1 = True
        for i in range(len(tag_names_BS2)):
            if detections_BS2[i][0] == tag:
                timestamps_BS2 = np.array(detections_BS2[i][1:])
                BS2 = True
        for i in range(len(tag_names_BS3)):
            if detections_BS3[i][0] == tag:
                timestamps_BS3 = np.array(detections_BS3[i][1:])
                BS3 = True
        for i in range(len(tag_names_BS4)):
            if detections_BS4[i][0] == tag:
                timestamps_BS4 = np.array(detections_BS4[i][1:])
                BS4 = True
        
        if BS1 == True and BS2 == True and BS3 == True and BS4 == True:
            next
        else:
            continue
        
        #To maintain nomenclature
        timestamps_BS1_i = timestamps_BS1 + 0.
        timestamps_BS2_i = timestamps_BS2 + 0.
        timestamps_BS3_i = timestamps_BS3 + 0.
        timestamps_BS4_i = timestamps_BS4 + 0.
        
        #Correct the clocks drifft (SLOPE,OFFSET): corese adjust using liner curve fitting
        timestamps_BS1_d0 = timestamps_BS1_i + 0.
        timestamps_BS2_d0 = timestamps_BS2_i + (timestamps_BS2_i*SLOPE[0]+OFFSET[0])
        timestamps_BS3_d0 = timestamps_BS3_i + (timestamps_BS3_i*SLOPE[1]+OFFSET[1])
        timestamps_BS4_d0 = timestamps_BS4_i + (timestamps_BS4_i*SLOPE[2]+OFFSET[2])
        
        #Correct the clocks drifft: fine adjust using Python cureve fitting    
        timestamps_BS1_d1 = timestamps_BS1_d0 + 0.
        timestamps_BS2_d1 = timestamps_BS2_d0 + clock_drifft_adjust(timestamps_BS2_d0,pop_12)
        timestamps_BS3_d1 = timestamps_BS3_d0 + clock_drifft_adjust(timestamps_BS3_d0,pop_13)
        timestamps_BS4_d1 = timestamps_BS4_d0 + clock_drifft_adjust(timestamps_BS4_d0,pop_14)
        
        #Correct the clocks drifft: ultrafine adjust using polinomial curve fitting by windowing
        timestamps_BS1_d2 = timestamps_BS1_d1 + 0.
        timestamps_BS2_d2 = np.array([])
        timestamps_BS3_d2 = np.array([])
        timestamps_BS4_d2 = np.array([])
        low = 0
        high = 0
        for i in range(num_splits):
            # find the upper time limit to change the fitting parameters
            while (True):
                if timestamps_BS2_d1[high] >= x1_s[i][-1] or high == timestamps_BS2_d1.size-1:
                    break
                high += 1
            aux_BS2 = timestamps_BS2_d1[low:high] + clock_drifft_adjust_s(timestamps_BS2_d1[low:high],np.array(pop_12_s[i]))
            timestamps_BS2_d2 = np.concatenate((timestamps_BS2_d2,aux_BS2),axis=0)
            low = high + 0
        low = 0
        high = 0
        for i in range(num_splits):
            # find the upper time limit to change the fitting parameters
            while (True):
                if timestamps_BS3_d1[high] >= x1_s[i][-1] or high == timestamps_BS3_d1.size-1:
                    break
                high += 1
            aux_BS3 = timestamps_BS3_d1[low:high] + clock_drifft_adjust_s(timestamps_BS3_d1[low:high],np.array(pop_13_s[i]))
            timestamps_BS3_d2 = np.concatenate((timestamps_BS3_d2,aux_BS3),axis=0)
            low = high + 0
        low = 0
        high = 0
        for i in range(num_splits):
            # find the upper time limit to change the fitting parameters
            while (True):
                if timestamps_BS4_d1[high] >= x1_s[i][-1] or high == timestamps_BS4_d1.size-1:
                    break
                high += 1
            aux_BS4 = timestamps_BS4_d1[low:high] + clock_drifft_adjust_s(timestamps_BS4_d1[low:high],np.array(pop_14_s[i]))
            timestamps_BS4_d2 = np.concatenate((timestamps_BS4_d2,aux_BS4),axis=0)
            low = high + 0   
        
        #Correct the clocks based on the computed time differences bwetween two pairs of syncTAG/BaseStations
        timestamps_BS1_d = timestamps_BS1_d2 + 0.
        timestamps_BS2_d = timestamps_BS2_d2 + dif_12_21
        timestamps_BS3_d = timestamps_BS3_d2 + dif_13_31
        timestamps_BS4_d = timestamps_BS4_d2 + dif_14_41
        
        #Starting the timestamp equalt to 0 in order to eliminate possible errors due to too big times
        try:
            aux1 = timestamps_BS1_d[0]
        except:
            aux1 = 0.
        try:
            aux2 = timestamps_BS2_d[0]
        except:
            aux2 = 0.
        try:
            aux3 = timestamps_BS3_d[0]
        except:
            aux3 = 0.
        try:
            aux4 = timestamps_BS4_d[0]
        except:
            aux4 = 0.
        t_offset = np.max(np.array([aux1,aux2,aux3,aux4]))
        timestamps_BS1_d -= t_offset
        timestamps_BS2_d -= t_offset
        timestamps_BS3_d -= t_offset
        timestamps_BS4_d -= t_offset
        
        #save the tag detection time of each base station with the corrected timestamp
        if os.path.exists(SAVE_FOLDER+'_BS1') == False:
            os.mkdir(SAVE_FOLDER+'_BS1')
        if os.path.exists(SAVE_FOLDER+'_BS2') == False:
            os.mkdir(SAVE_FOLDER+'_BS2')
        if os.path.exists(SAVE_FOLDER+'_BS3') == False:
            os.mkdir(SAVE_FOLDER+'_BS3')
        if os.path.exists(SAVE_FOLDER+'_BS4') == False:
            os.mkdir(SAVE_FOLDER+'_BS4')
#        np.savetxt(SAVE_FOLDER+'_BS1/detections_BS1_'+tag+'.txt', np.append(np.matrix(timestamps_BS1_d+t_offset).T,np.matrix(np.repeat(tag,timestamps_BS1_d.size)).T,axis=1),delimiter=',',header='Tag ID = ' + tag + ', Timestamp(s), TagID', fmt="%s")
#        np.savetxt(SAVE_FOLDER+'_BS2/detections_BS2_'+tag+'.txt', np.append(np.matrix(timestamps_BS2_d+t_offset).T,np.matrix(np.repeat(tag,timestamps_BS2_d.size)).T,axis=1),delimiter=',',header='Tag ID = ' + tag + ', Timestamp(s), TagID', fmt="%s")
#        np.savetxt(SAVE_FOLDER+'_BS3/detections_BS3_'+tag+'.txt', np.append(np.matrix(timestamps_BS3_d+t_offset).T,np.matrix(np.repeat(tag,timestamps_BS3_d.size)).T,axis=1),delimiter=',',header='Tag ID = ' + tag + ', Timestamp(s), TagID', fmt="%s")
#        np.savetxt(SAVE_FOLDER+'_BS4/detections_BS4_'+tag+'.txt', np.append(np.matrix(timestamps_BS4_d+t_offset).T,np.matrix(np.repeat(tag,timestamps_BS4_d.size)).T,axis=1),delimiter=',',header='Tag ID = ' + tag + ', Timestamp(s), TagID', fmt="%s")

        np.savetxt(SAVE_FOLDER+'_BS1/detections_BS1_'+tag+'.txt', timestamps_BS1_d+t_offset,delimiter=',',header='Tag ID = ' + tag + ', Timestamp(s)', fmt="%.9f")
        np.savetxt(SAVE_FOLDER+'_BS2/detections_BS2_'+tag+'.txt', timestamps_BS2_d+t_offset,delimiter=',',header='Tag ID = ' + tag + ', Timestamp(s)', fmt="%.9f")
        np.savetxt(SAVE_FOLDER+'_BS3/detections_BS3_'+tag+'.txt', timestamps_BS3_d+t_offset,delimiter=',',header='Tag ID = ' + tag + ', Timestamp(s)', fmt="%.9f")
        np.savetxt(SAVE_FOLDER+'_BS4/detections_BS4_'+tag+'.txt', timestamps_BS4_d+t_offset,delimiter=',',header='Tag ID = ' + tag + ', Timestamp(s)', fmt="%.9f")
  

       
        #Computing the time difference between receptions
        a = 0
        b = 0
        c = 0
        d = 0
        threshold = 10
        detection_number = 0
        t_position_t = []
        t_position_t_mle = []
        time_res =  []
        time_merge = []
        while (True):
#            if detection_number == 10:
#                break
            t_min = 1e12
            try:
                if timestamps_BS1_d[a] < t_min:
                    t_min = timestamps_BS1_d[a]
            except:
                next
            try:
                if timestamps_BS2_d[b] < t_min:
                    t_min = timestamps_BS2_d[b]
            except:
                next
            try:
                if timestamps_BS3_d[c] < t_min:
                    t_min = timestamps_BS3_d[c]
            except:
                next
            try:
                if timestamps_BS4_d[d] < t_min:
                    t_min = timestamps_BS4_d[d]
            except:
                next
            if t_min == 1e12:
                break
            rt_aux = []
            BS_aux = []
            BS1 = False
            BS2 = False
            BS3 = False
            BS4 = False
            try:
                if timestamps_BS1_d[a] >= t_min-threshold and timestamps_BS1_d[a] <= t_min+threshold:
                    rt_aux.append(timestamps_BS1_d[a])
                    BS1 = True
                    BS_aux.append(A1)
                    a += 1
            except:
                next
            try:
                if timestamps_BS2_d[b] >= t_min-threshold and timestamps_BS2_d[b] <= t_min+threshold:
                    rt_aux.append(timestamps_BS2_d[b])
                    BS2 = True
                    BS_aux.append(A2)
                    b += 1
            except:
                next
            try:
                if timestamps_BS3_d[c] >= t_min-threshold and timestamps_BS3_d[c] <= t_min+threshold:
                    rt_aux.append(timestamps_BS3_d[c])
                    BS3 = True
                    BS_aux.append(A3)
                    c += 1
            except:
                next
            try:
                if timestamps_BS4_d[d] >= t_min-threshold and timestamps_BS4_d[d] <= t_min+threshold:
                    rt_aux.append(timestamps_BS4_d[d])
                    BS4 = True
                    BS_aux.append(A4)
                    d += 1
            except:
                next
            
            #save a merge tag timestamp detection from any base station
            rt = np.array(rt_aux)
            BS = np.array(BS_aux)
            if rt.size > 0:
                time_merge.append(rt.item(0)+t_offset)
        
        #save marge tag detection from at least 1 base station
        if os.path.exists(SAVE_FOLDER+'_BSM') == False:
            os.mkdir(SAVE_FOLDER+'_BSM')
        np.savetxt(SAVE_FOLDER+'_BSM/detections_merge_'+tag+'.txt', np.append(np.matrix(time_merge).T,np.matrix(np.repeat(tag,len(time_merge))).T,axis=1),delimiter=',',header='Tag ID = ' + tag + ', Timestamp(s), TagID', fmt="%s")
              

#%%
print '###############################################################'
print 'The following tags has been detected:'
print tag_names_total
print '###############################################################'

        
#%%
#Plot Base station
ls = 12
fig = plt.figure(figsize=(5,5))
plt.plot(A1_INITIAL[0],A1_INITIAL[1],'bs',ms=10,lw=1,label = 'Initial')
plt.plot(A2_INITIAL[0],A2_INITIAL[1],'bs',ms=10,lw=1)
plt.plot(A3_INITIAL[0],A3_INITIAL[1],'bs',ms=10,lw=1)
plt.plot(A4_INITIAL[0],A4_INITIAL[1],'bs',ms=10,lw=1)
plt.text(A1_INITIAL[0]+5,A1_INITIAL[1],'BS(A)',size=ls)
plt.text(A2_INITIAL[0]+5,A2_INITIAL[1],'BS(B)',size=ls)
plt.text(A3_INITIAL[0]+5,A3_INITIAL[1],'BS(D)',size=ls)
plt.text(A4_INITIAL[0]+5,A4_INITIAL[1],'BS(E)',size=ls)

plt.plot(A1[0],A1[1],'r^',ms=10,lw=1,alpha = 0.5,label = 'Computed')
plt.plot(A2[0],A2[1],'r^',ms=10,lw=1,alpha = 0.5)
plt.plot(A3[0],A3[1],'r^',ms=10,lw=1,alpha = 0.5)
plt.plot(A4[0],A4[1],'r^',ms=10,lw=1,alpha = 0.5)

#angle = -3
#angle = angle*np.pi/180.
#plt.plot(A1[0],A1[1],'y^',ms=10,lw=1,alpha = 0.5,label = 'rotated')
#plt.plot((A2[0]-A1[0])*np.cos(angle)-(A2[1]-A1[1])*np.sin(angle)+A1[0],(A2[0]-A1[0])*np.sin(angle)+(A2[1]-A1[1])*np.cos(angle)+A1[1],'y^',ms=10,lw=1,alpha = 0.5)
#plt.plot((A3[0]-A1[0])*np.cos(angle)-(A3[1]-A1[1])*np.sin(angle)+A1[0],(A3[0]-A1[0])*np.sin(angle)+(A3[1]-A1[1])*np.cos(angle)+A1[1],'y^',ms=10,lw=1,alpha = 0.5)
#plt.plot((A4[0]-A1[0])*np.cos(angle)-(A4[1]-A1[1])*np.sin(angle)+A1[0],(A4[0]-A1[0])*np.sin(angle)+(A4[1]-A1[1])*np.cos(angle)+A1[1],'y^',ms=10,lw=1,alpha = 0.5)


plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='major',width = 1, length=5)
plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='minor',width = 0.75, length=2.5)
plt.axes().xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.axes().yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.xticks(fontsize=ls)
plt.yticks(fontsize=ls)

plt.locator_params(axis='x', nbins=3)
plt.xlabel('Easting (m)',size=ls)
plt.ylabel('Northing (m)',size=ls)
plt.axis('equal')

plt.legend()

plt.savefig(SAVE_FOLDER+'_moorings.jpg',format='jpg', dpi=800 ,bbox_inches='tight')
plt.show()       
print ''
