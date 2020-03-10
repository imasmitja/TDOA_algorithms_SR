# -*- coding: utf-8 -*-
"""
Created on Wed Dec 04 07:57:40 2019

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



#Acoustic receiver localizations (RESNEP) reals
A1_gps = np.array([543979.91,4651542.64,30.])
A2_gps = np.array([544117.92,4651541.96,25.])
A3_gps = np.array([543980.73,4651411.57,20.])
A4_gps = np.array([544132.64,4651394.00,20.])

A1_usbl = np.array([543957.71,4651491.78,30.])
A2_usbl = np.array([544094.56,4651484.2,25.])
A3_usbl = np.array([543955.9,4651359.9,20.])
A4_usbl = np.array([544110.4,4651336.7,20.])


#%%
#Compute the new Vemco positions based on distances between pairs of BS. This has been computed using the
#time of flight of sync tags
d_12 = 131.83697853379576
d_13 = 142.64290299956545
d_14 = 215.57350329792817
d_23 = 191.61799518459424
d_24 = 142.78241237925238

h_3 = np.sqrt(d_23**2-(d_12**2+d_23**2-d_13**2)**2/(4*d_12**2))
x_3 = np.sqrt(d_13**2-h_3**2)

h_4 = np.sqrt(d_24**2-(d_12**2+d_24**2-d_14**2)**2/(4*d_12**2))
x_4 = np.sqrt(d_14**2-h_4**2)

A1_new = np.array([0,0,0.])
A2_new = np.array([d_12,0.,0.])
A3_new = np.array([x_3,h_3,0.])
A4_new = np.array([x_4,h_4,0.])

#sync tag positions computed using BS
#sync_A = np.array([543979.88133204367,4651541.1230002521])
#sync_B = np.array([544110.89004007343,4651539.5480907783])
#sync_C = np.array([544120.48638093681,4651495.0315945074])
#sync_D = np.array([543982.86808028747,4651399.2920466028])
#sync_E = np.array([544143.72974668595,4651403.6927831052])

#New ones using the BS positions provided by the ROV's USBL
sync_A = np.array([543957.89390830451,4651491.828311556])
sync_B = np.array([544088.79368197126,4651491.5078131091])
sync_C = np.array([544096.29597412364,4651444.6060652174])
sync_D = np.array([543966.74440833204,4651353.2306500096])
sync_E = np.array([544112.17127613176,4651358.0134112304])

#sync tag positions computed using BS_new


#Plot the tag's track
ls = 12
fig = plt.figure(figsize=(5,5))
plt.plot(A1_gps[0],A1_gps[1],'bx',ms=10,lw=1,label = 'GPS')
plt.plot(A2_gps[0],A2_gps[1],'bx',ms=10,lw=1)
plt.plot(A3_gps[0],A3_gps[1],'bx',ms=10,lw=1)
plt.plot(A4_gps[0],A4_gps[1],'bx',ms=10,lw=1)
plt.plot(544058,4651449,'bx',ms=10,lw=1)
plt.text(A1_gps[0]+5,A1_gps[1],'BS(A)',size=ls)
plt.text(A2_gps[0]+5,A2_gps[1],'BS(B)',size=ls)
plt.text(A3_gps[0]+5,A3_gps[1],'BS(D)',size=ls)
plt.text(A4_gps[0]+5,A4_gps[1],'BS(E)',size=ls)
plt.text(544058+5,4651449,'C',size=ls) 


plt.plot(A1_usbl[0],A1_usbl[1],'rx',ms=10,lw=1,label = 'USBL')
plt.plot(A2_usbl[0],A2_usbl[1],'rx',ms=10,lw=1)
plt.plot(A3_usbl[0],A3_usbl[1],'rx',ms=10,lw=1)
plt.plot(A4_usbl[0],A4_usbl[1],'rx',ms=10,lw=1)
plt.plot(544097.75,4651435.64,'rx',ms=10,lw=1)
plt.text(A1_usbl[0]+5,A1_usbl[1],'BS(A)',size=ls)
plt.text(A2_usbl[0]+5,A2_usbl[1],'BS(B)',size=ls)
plt.text(A3_usbl[0]+5,A3_usbl[1],'BS(D)',size=ls)
plt.text(A4_usbl[0]+5,A4_usbl[1],'BS(E)',size=ls)
plt.text(544097.75+5,4651435.64,'C',size=ls) 

plt.plot(A1_usbl[0]+A1_new[0],A1_usbl[1]-A1_new[1],'r^',ms=10,lw=1,alpha = 0.5,label = 'USBL+TOF')
plt.plot(A1_usbl[0]+A2_new[0],A1_usbl[1]-A2_new[1],'r^',ms=10,lw=1,alpha = 0.5)
plt.plot(A1_usbl[0]+A3_new[0],A1_usbl[1]-A3_new[1],'r^',ms=10,lw=1,alpha = 0.5)
plt.plot(A1_usbl[0]+A4_new[0],A1_usbl[1]-A4_new[1],'r^',ms=10,lw=1,alpha = 0.5)

plt.plot(sync_A[0],sync_A[1],'yo',ms=15,fillstyle='none',alpha = 0.7,label = 'TDOF')
plt.plot(sync_B[0],sync_B[1],'yo',ms=15,fillstyle='none',alpha = 0.7)
plt.plot(sync_C[0],sync_C[1],'yo',ms=15,fillstyle='none',alpha = 0.7)
plt.plot(sync_D[0],sync_D[1],'yo',ms=15,fillstyle='none',alpha = 0.7)
plt.plot(sync_E[0],sync_E[1],'yo',ms=15,fillstyle='none',alpha = 0.7)


#plt.plot(543975,4649950,'bo',ms=10,lw=1,alpha=0.5,label = 'tag_ROV')
#plt.text(543975+10,4649950,'tag_ROV',size=ls) 

#plt.plot(543872.25,4651416.11,'bo',ms=10,lw=1,alpha=0.5, label='G500')
#plt.text(543872.25+10,4651416.11,'G500_circle',size=ls) 

plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='major',width = 1, length=5)
plt.tick_params(direction='in',bottom=True,top=True,left=True,right=True,which='minor',width = 0.75, length=2.5)
plt.axes().xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.axes().yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.xticks(fontsize=ls)
plt.yticks(fontsize=ls)

plt.locator_params(axis='x', nbins=3)
#plt.title('Tag ID: '+ tag,size=ls)
plt.xlabel('Easting (m)',size=ls)
plt.ylabel('Northing (m)',size=ls)
plt.axis('equal')

plt.legend(ncol=1,bbox_to_anchor=(1.35, 1.02))

plt.savefig('RESNEP_moorings.jpg',format='jpg', dpi=800 ,bbox_inches='tight')
plt.show()       
print ''