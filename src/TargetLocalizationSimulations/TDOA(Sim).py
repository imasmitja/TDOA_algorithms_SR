# -*- coding: utf-8 -*-
"""
Created on Fri March 13 14:10:49 2020

@author: Ivan Masmitja
"""

"""
This code have been obtained from: https://salzis.wordpress.com/2015/05/25/particle-filters-with-python/

https://www.udacity.com/course/viewer#!/c-cs373/l-48704330/m-48665972

Particle filters with Python
Particle filters comprise a broad family of Sequential Monte Carlo (SMC) algorithms for 
approximate inference in partially observable Markov chains. The objective of a particle 
filter is to estimate the posterior density of the state variables given the observation 
variables. A generic particle filter estimates the posterior distribution of the hidden
 states using the observation measurement process.
 
It was used previously to compute target tracking using range-only methods. 
Now it uses the TDOA of a signal using 4 beacons as base station (BS) or landmarks

This has been conducted to present the results at Science Robotics journal and
to compute the data obtained during the RESNEP campaing

The algorithms used are:
    1- Maximum Likelihood (ML) estimation
    2- Close-form Weighted Leas Square (WLS)
    3- Particle Filter (PF)
    4- Maximum A Posteriori (MAP) estimation
    5- A combination between WLS and ML (WLS-ML)
    6- MAP with marginalizing states (MAP-M)

"""

#In this example the target lives in a 2-dimensional world.
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import cos
from numpy import sin
from numpy import sqrt
from numpy import pi
import datetime
import matplotlib.patches as mpatches
from scipy import interpolate
import time
from TargetClass_SR import target_class
import calendar
import matplotlib.ticker as ticker

########################################################################################
########################################################################################

#Simulation parameters
iteration_number = 1 #steps
toa_noise_sd_t = [0.0005,0.001,0.0015,0.002] #Noise added to the toa measurements (z)
toa_noise_sd_t = [0.001] #Noise added to the toa measurements (z)
toa_noise_outlier = False #if True we add 0.1 s on rt[2] at measurement number 7

printplot = True

track_from_file = True # if False == random walk is simulated

#np.random.seed(12345)

######################################################################################################
# Functions
######################################################################################################


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
            timestamp_BS_s.append(calendar.timegm(time.strptime(t[:-7], '%Y-%m-%d %H:%M:%S')))
            timestamp_BS_ms.append(float(t[-7:]))
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

#def clock_drifft_adjust(x, pop):
#    aux = np.ones(x.size)
#    a = pop.item(0)
#    b = pop.item(1)
#    c = pop.item(2)
#    d = pop.item(3)
#    for i in range(x.size):
#        aux[i] = a*x[i]**3 + b*x[i]**2 +c*x[i] + d
#    return  aux
#
#def clock_drifft_adjust_2(x, pop):
#    aux = np.ones(x.size)
#    a = pop.item(0)
#    b = pop.item(1)
#    for i in range(x.size):
#        aux[i] = a*x[i] + b
#    return  aux
    
def random_track(A1,A2,A3,A4):
    #Target position (simulated)
    target_velocity = 0.2 #m/s
    dt = 60 #seconds
    num_points = 85
    turn_points = [15,35,52]
    turn_angles = [-90.,-90.,-90.] #degrees
    distance = target_velocity*dt
    new_x = 50.
    new_y = 20.
    angle = 90.*np.pi/180.
    n=0
    random_walk = True
    random_angle = 5. #in degree
    #np.random.seed(24123)
    
    only_random = True  
    num_points = 60
    random_angle = 50. #in degree
    #np.random.seed(2234)
    
    cs=1500.
    target_t = []
    t1t_t = []
    t2t_t = []
    t3t_t = []
    t4t_t = []
    for i in range(num_points):
        target = np.array([new_x,new_y,0.])
        target_t.append(target)
        #Range (simulated)
        r1t = np.sqrt(np.matrix((target-A1))*np.matrix((target-A1)).T).item(0)
        r2t = np.sqrt(np.matrix((target-A2))*np.matrix((target-A2)).T).item(0)
        r3t = np.sqrt(np.matrix((target-A3))*np.matrix((target-A3)).T).item(0)
        r4t = np.sqrt(np.matrix((target-A4))*np.matrix((target-A4)).T).item(0)
        #Time of arrival at each detector in seconds
        t1t = r1t/cs+dt*i
        t2t = r2t/cs+dt*i
        t3t = r3t/cs+dt*i
        t4t = r4t/cs+dt*i
        t1t_t.append(t1t)
        t2t_t.append(t2t)
        t3t_t.append(t3t)
        t4t_t.append(t4t)
        
        #move the target
        if only_random == True:
            if random_walk == True:
                random_a = np.random.rand()*random_angle*2-random_angle
                angle += random_a*np.pi/180.
        else:
            if i == turn_points[n]:
                angle += turn_angles[n]*np.pi/180.
                n+=1
                if n == len(turn_points):
                    n-=1
            if random_walk == True:
                random_a = np.random.rand()*random_angle*2-random_angle
                angle += random_a*np.pi/180.
            
        new_x += distance*np.cos(angle)
        new_y += distance*np.sin(angle)
        
        
    target_t = np.matrix(target_t)
    target_t = np.array(target_t.T[0:2].T)
    actual_time = time.time()
    timestamps_BS1_d = np.array(t1t_t) + actual_time
    timestamps_BS2_d = np.array(t2t_t) + actual_time
    timestamps_BS3_d = np.array(t3t_t) + actual_time
    timestamps_BS4_d = np.array(t4t_t) + actual_time
    return(timestamps_BS1_d,timestamps_BS2_d,timestamps_BS3_d,timestamps_BS4_d,target_t)

def visualization(t_position_t,A1,A2,A3,A4,step,t1,printplot):   
    #draw particles
    if printplot == True:
        plt.figure(figsize=(5,5))
        plt.title('Filter, step ' + str(step))
        plt.plot(A1[0],A1[1],'rx',ms=10,lw=1)
        plt.plot(A2[0],A2[1],'rx',ms=10,lw=1)
        plt.plot(A3[0],A3[1],'rx',ms=10,lw=1)
        plt.plot(A4[0],A4[1],'rx',ms=10,lw=1)

    targetrealx = np.array(np.matrix(t1.txs).T[0])[0]
    targetrealy = np.array(np.matrix(t1.txs).T[2])[0]
    
    particleprint = False 
    if particleprint == True:
        #draw old particles
        xx = t1.pf.x_old.T[0]
        yy = t1.pf.x_old.T[2]
        plt.plot(xx,yy,'ro',alpha=0.1,ms=10)
        #draw resampled particles
        xx = t1.pf.x.T[0]
        yy = t1.pf.x.T[2]
        plt.plot(xx,yy,'bo',alpha=0.1,ms=1)

    if printplot == True:
        #Print lines between true target position and its estimations
        plt.plot([t1.txs[-1][0],t1.pxs[-1][0]], [t1.txs[-1][2],t1.pxs[-1][2]],'k--',alpha=0.25)
        plt.plot([t1.txs[-1][0],t1.mmxs[-1][0]], [t1.txs[-1][2],t1.mmxs[-1][2]],'k--',alpha=0.25)
        plt.plot([t1.txs[-1][0],t1.lsxs[-1][0]], [t1.txs[-1][2],t1.lsxs[-1][2]],'k--',alpha=0.25)
        
        # target's location
        arrow = plt.arrow(t1.txs[-1][0], t1.txs[-1][2],2*t1.txs[-1][1],2*t1.txs[-1][3],shape='full',lw=1,length_includes_head=False,head_width=20, facecolor='#6666ff', edgecolor='#0000cc')
        plt.gca().add_patch(arrow)

        #print target position estimation
        arrow = plt.arrow(t1.pxs[-1][0],t1.pxs[-1][2],2*cos(t1.pxs[-1][4]),2*sin(t1.pxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='#fff0f5',edgecolor='#0000cc',alpha=0.4)
        plt.gca().add_patch(arrow) 

        #print target position estimation using MAP
        arrow = plt.arrow(t1.mxs[-1][0],t1.mxs[-1][2],2*cos(t1.mxs[-1][4]),2*sin(t1.mxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='m',edgecolor='#0000cc',alpha=0.4)
        plt.gca().add_patch(arrow)
        
         #print target position estimation using MAP
        arrow = plt.arrow(t1.mmxs[-1][0],t1.mmxs[-1][2],2*cos(t1.mmxs[-1][4]),2*sin(t1.mmxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='#90ee90',edgecolor='#0000cc',alpha=0.4)
        plt.gca().add_patch(arrow)
           
        #print target position estimation using LS
        arrow = plt.arrow(t1.lsxs[-1][0],t1.lsxs[-1][2],2*cos(t1.lsxs[-1][4]),2*sin(t1.lsxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='yellow',edgecolor='#0000cc',alpha=0.4)
        plt.gca().add_patch(arrow)
        
        #print target position estimation using ML
        arrow = plt.arrow(t1.mlxs[-1][0],t1.mlxs[-1][2],2*cos(t1.mlxs[-1][4]),2*sin(t1.mlxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='green',edgecolor='#0000cc',alpha=0.4)
        plt.gca().add_patch(arrow)
        
        #print target position estimation using LSML
        arrow = plt.arrow(t1.lsmlxs[-1][0],t1.lsmlxs[-1][2],2*cos(t1.lsmlxs[-1][4]),2*sin(t1.lsmlxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='b',edgecolor='#0000cc',alpha=0.4)
        plt.gca().add_patch(arrow)
                 
        wg_patch = mpatches.Patch(facecolor='red', label='$WG$',linewidth=1,edgecolor='black')
        t_patch = mpatches.Patch(facecolor='blue', label='$T$',linewidth=1,edgecolor='black')
        pf_patch = mpatches.Patch(facecolor='#fff0f5', label='$\overline{PF}$',linewidth=1,edgecolor='black')
        map_patch = mpatches.Patch(facecolor='m', label='$\overline{MAP}$',linewidth=1,edgecolor='black')
        map2_patch = mpatches.Patch(facecolor='#90ee90', label='$\overline{MAP-M}$',linewidth=1,edgecolor='black')
        ls_patch = mpatches.Patch(facecolor='yellow', label='$\overline{WLS}$',linewidth=1,edgecolor='black')
        ml_patch = mpatches.Patch(facecolor='green', label='$\overline{ML}$',linewidth=1,edgecolor='black')
        lsml_patch = mpatches.Patch(facecolor='b', label='$\overline{WLS-ML}$',linewidth=1,edgecolor='black')
        plt.legend(handles=[wg_patch,t_patch,pf_patch,map_patch,map2_patch,ls_patch,ml_patch,lsml_patch],ncol=1,bbox_to_anchor=(1.4, 1.02))                
        
        plt.plot(targetrealx,targetrealy,'b-',alpha=0.5)
        plt.plot(np.array(np.matrix(t1.pxs).T[0])[0],np.array(np.matrix(t1.pxs).T[2])[0],'m*--',color='#fff0f5',alpha=0.3)
        plt.plot(np.array(np.matrix(t1.mxs).T[0])[0],np.array(np.matrix(t1.mxs).T[2])[0],'m*--',alpha=0.3)
        plt.plot(np.array(np.matrix(t1.mmxs).T[0])[0],np.array(np.matrix(t1.mmxs).T[2])[0],'c*--',color='#90ee90',alpha=0.3)
        plt.plot(np.array(np.matrix(t1.lsxs).T[0])[0],np.array(np.matrix(t1.lsxs).T[2])[0],'y*--',alpha=0.3)
        plt.plot(np.array(np.matrix(t1.mlxs).T[0])[0],np.array(np.matrix(t1.mlxs).T[2])[0],'g*--',alpha=0.3)
        plt.plot(np.array(np.matrix(t1.lsmlxs).T[0])[0],np.array(np.matrix(t1.lsmlxs).T[2])[0],'b*--',alpha=0.3)
        
    targetrealx = np.array(np.matrix(t1.txs).T[0])[0]
    targetrealy = np.array(np.matrix(t1.txs).T[2])[0]  
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.axis('equal')
    plt.xlim(-300,300)
    plt.ylim(-300,300)
    
#    fsave=plt.gcf()
#    fsave.set_size_inches(6,6)
#    plt.tight_layout()
#    fsave.autolayout=True
#    plt.savefig("figure_9T_6262018_" + str(step) + ".jpg",dpi=80,bbox_inches='tight')
    
    plt.show()
    return()




#%%
########################################################################################
# START MAIN PROGRAM 
########################################################################################

for toa_noise_sd in toa_noise_sd_t:
    # Initialize variables
    terror=[]
    terror2=[]
    tyerror = []
    txerror = []
    xlimit=[0,0]
    xold=[-100,400]
    ylimit=[0,0]
    yold=[-100,400]
    old_mywg = [0.,0.]
    targetrealx=[]
    targetrealy=[]
    targetpredictionx_pf=[]
    targetpredictiony_pf=[]
    targetpredictionx_ekf=[]
    targetpredictiony_ekf=[]
    targetpredictionx_ukf=[]
    targetpredictiony_ukf=[]
    targetpredictionx_map=[]
    targetpredictiony_map=[]
    targetpredictionx_map2=[]
    targetpredictiony_map2=[]
    wgrealx=[]
    wgrealy=[]
    wgrangex=[]
    wgrangey=[]
    angle_wgtp_old = 0
    new_circle = True
    angle=-np.pi/2
    tangle=[]
    wg_velocity_append=[]
    
    angle1=0
    angle2=0    
    
    testx1=[]
    testx2=[]
    testx3=[]    
    
    ep_pf = []
    ep_map = []
    ep_map2 = []
    ep_ml = []
    ep_lsml = []
    ep_ls = []
    t_pf = []
    t_map = []
    t_map2 = []
    t_ml = []
    t_lsml = []
    t_ls = []
    dt = 0.01
        
    for it_num in range(iteration_number): 
        ###############################################################################
        ###                  Parameters initialization                             ####
        ###############################################################################
        #Create Observer
        oxs = []
        myobserver = np.array([0.,0.,0.,0.])   
        
        #Create targets
        print "Initializing Target's Estimations..."
        mytarget = np.array([0.,0.,0.,0.0])
        t1=target_class(mytarget,myobserver)
        print "Targets created"
    
        #Time buffer constants
        pft=[]
        ekft=[]
        ukft=[]
        mapt=[]
        map2t=[]
        lst=[]
        mlt=[]
        lsmlt=[]
        
        #Initial mytarget angle
        old_angle = pi/2.

        #if outlier we compute where to put it
        rand_itnum = int(np.random.rand()*18)
        rand_tofnum = int(np.random.rand()*3)
        
        #read Vemco file (.csv) for each Base station (BS)
        filename_BS1 = 'VemcoFiles\TestFile_31b.csv'
        filename_BS2 = 'VemcoFiles\TestFile_32b.csv'
        filename_BS3 = 'VemcoFiles\TestFile_33b.csv'
        filename_BS4 = 'VemcoFiles\TestFile_34b.csv'
        
        t_position_real = np.array([[0.,0.],
                                [0.,60.],
                                [0.,120.],
                                [0.,180.],
                                [60.,180.],
                                [120.,180.],
                                [180.,180.],
                                [240.,180.],
                                [240.,120.],
                                [240.,60.],
                                [240.,0.],
                                [240.,-60.],
                                [180.,-60.],
                                [120.,-60.],
                                [60.,-60.],
                                [0.,-60.],
                                [-60.,-60.],
                                [-120.,-60.],
                                [-180.,-60.]])
        
        ###Acoustic receiver localizations
        A1 = np.array([100.,100.,0.])
        A2 = np.array([-100.,100.,0.])
        A3 = np.array([-100.,-100.,0.])
        A4 = np.array([100.,-100.,0.])
        
        if track_from_file == True:
            #read the viles created to simulate a target moveing at 1 m/s and transmiting a signal every 60 s
            detections_BS1,tag_names_BS1 = read_BS_file(filename_BS1)
            detections_BS2,tag_names_BS2 = read_BS_file(filename_BS2)
            detections_BS3,tag_names_BS3 = read_BS_file(filename_BS3)
            detections_BS4,tag_names_BS4 = read_BS_file(filename_BS4)
            
            timestamps_BS1_d = np.array(detections_BS1[0][1:]) +0.
            timestamps_BS2_d = np.array(detections_BS2[0][1:]) +0.
            timestamps_BS3_d = np.array(detections_BS3[0][1:]) +0.
            timestamps_BS4_d = np.array(detections_BS4[0][1:]) +0.
        
        else:
            #generate a simulated tag moving randomly
            timestamps_BS1_d,timestamps_BS2_d,timestamps_BS3_d,timestamps_BS4_d,t_position_real = random_track(A1,A2,A3,A4)    
     
        #Starting the timestamp equalt to 0 in order to eliminate possible errors due to too big times
        t_offset = int(np.max(np.array([timestamps_BS1_d[0],timestamps_BS2_d[0],timestamps_BS3_d[0],timestamps_BS4_d[0]])))
        timestamps_BS1_d -= t_offset
        timestamps_BS2_d -= t_offset
        timestamps_BS3_d -= t_offset
        timestamps_BS4_d -= t_offset
        
        #Computing the time difference between receptions
        a = 0
        b = 0
        c = 0
        d = 0
        threshold = 10
        detection_number = 0
        t_position_t = []
        time_res =  []
        old_rt = -1
        n = 0
        timet_aux = []
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
                
            RT = np.array(rt_aux)
            BS = np.array(BS_aux)
            
            #Save the true target position 
            if RT.size > 2:
                t_position = t_position_real[detection_number]
                t_position_t.append(t_position)
                time_res.append(RT.item(0)+t_offset)
            detection_number += 1
                
            ###############################################################################
            ###                        Compute positions                                ###
            ##############################################################################
            # 1- move the target 
            #New Target position
            mytarget_old = t1.mytarget
            try:
                vx = (t_position.item(0)-t1.txs[-1][0])/dt
                vy = (t_position.item(1)-t1.txs[-1][2])/dt
                if vx == 0 and vy == 0:    
                    vx = t1.txs[-1][1]
                    vy = t1.txs[-1][3]
                t1.mytarget = np.array([t_position.item(0) , vx , t_position.item(1), vy])
            except:
                t1.mytarget = np.array([t_position.item(0), 0. ,t_position.item(1), 0.])
            t1.txs.append(t1.mytarget)  
            BS = np.array(BS_aux)
            
            # 2- Adquiere a new measurement
            #Set the dt
            if old_rt == -1:
                old_rt = RT.item(0) - 10.01
            dt = RT.item(0) - old_rt
            old_rt = RT.item(0)
            timet_aux.append(RT.item(0)+t_offset)
            #add noise and outliers to the Time of Arrival signals
            RT_noise = RT + 0.
            for i in range(RT.size):
                RT_noise[i] += np.random.normal(0.,toa_noise_sd)
            if toa_noise_outlier == True and detection_number == rand_itnum:
                RT_noise[rand_tofnum] += 0.1
        
            # 3- Compute the target estimation using different algorithms 
            pf_time   = t1.updatePF(dt,RT_noise+0.,BS+0.)
            map_time  = t1.updateMAP(dt,RT_noise+0.,BS+0.)
            map2_time = t1.updateMAPm(dt,RT_noise+0.,BS+0.)  
            ls_time   = t1.updateLS(dt,RT_noise+0.,BS+0.) 
            ml_time   = t1.updateML(dt,RT_noise+0.,BS+0.)
            lsml_time   = t1.updateLSMLE(dt,RT_noise+0.,BS+0.)

            #5- Visualitzation (Plot print)
            t_velocity = np.sqrt((t1.mytarget[0]-mytarget_old[0])**2+(t1.mytarget[2]-mytarget_old[2])**2)/dt 
            t = detection_number +0
            print 'Step # %d of %d. Iteration # %d of %d. Noise=%.4f'%(t,t_position_real.size/2,it_num,iteration_number,toa_noise_sd)
            if printplot == True:
                visualization(t_position_t,A1,A2,A3,A4,t,t1,printplot) #Function to print the plot and save variables
            pft.append(pf_time)
            mapt.append(map_time)
            map2t.append(map2_time)
            lst.append(ls_time)
            mlt.append(ml_time)
            lsmlt.append(lsml_time)
                    
            ####################################################################################
            ##            
            ####################################################################################
            
        #Compute position error
        targetrealx = np.array(np.matrix(t1.txs).T[0])[0]
        targetrealy = np.array(np.matrix(t1.txs).T[2])[0] 
        targetpredictionx_pf = np.array(np.matrix(t1.pxs).T[0])[0]
        targetpredictiony_pf = np.array(np.matrix(t1.pxs).T[2])[0]
        targetpredictionx_map = np.array(np.matrix(t1.mxs).T[0])[0]
        targetpredictiony_map = np.array(np.matrix(t1.mxs).T[2])[0]
        targetpredictionx_map2 = np.array(np.matrix(t1.mmxs).T[0])[0]
        targetpredictiony_map2 = np.array(np.matrix(t1.mmxs).T[2])[0]
        targetpredictionx_ls = np.array(np.matrix(t1.lsxs).T[0])[0]
        targetpredictiony_ls = np.array(np.matrix(t1.lsxs).T[2])[0]
        targetpredictionx_ml = np.array(np.matrix(t1.mlxs).T[0])[0]
        targetpredictiony_ml = np.array(np.matrix(t1.mlxs).T[2])[0]
        targetpredictionx_lsml = np.array(np.matrix(t1.lsmlxs).T[0])[0]
        targetpredictiony_lsml = np.array(np.matrix(t1.lsmlxs).T[2])[0]
        xe_pf = np.array(targetrealx)-np.array(targetpredictionx_pf)
        ye_pf = np.array(targetrealy)-np.array(targetpredictiony_pf)
        xe_map = np.array(targetrealx)-np.array(targetpredictionx_map)
        ye_map = np.array(targetrealy)-np.array(targetpredictiony_map)
        xe_map2 = np.array(targetrealx)-np.array(targetpredictionx_map2)
        ye_map2 = np.array(targetrealy)-np.array(targetpredictiony_map2)
        xe_ls = np.array(targetrealx)-np.array(targetpredictionx_ls)
        ye_ls = np.array(targetrealy)-np.array(targetpredictiony_ls)
        xe_ml = np.array(targetrealx)-np.array(targetpredictionx_ml)
        ye_ml = np.array(targetrealy)-np.array(targetpredictiony_ml)
        xe_lsml = np.array(targetrealx)-np.array(targetpredictionx_lsml)
        ye_lsml = np.array(targetrealy)-np.array(targetpredictiony_lsml)
        
        #save error
        ep_pf.append(np.sqrt(xe_pf**2+ye_pf**2))
        ep_map.append(np.sqrt(xe_map**2+ye_map**2))
        ep_map2.append(np.sqrt(xe_map2**2+ye_map2**2))
        ep_ls.append(np.sqrt(xe_ls**2+ye_ls**2))
        ep_ml.append(np.sqrt(xe_ml**2+ye_ml**2))
        ep_lsml.append(np.sqrt(xe_lsml**2+ye_lsml**2))
        #save process time
        t_pf.append(np.array(pft))
        t_map.append(np.array(mapt))
        t_map2.append(np.array(map2t))
        t_ls.append(np.array(lst))
        t_ml.append(np.array(mlt))
        t_lsml.append(np.array(lsmlt))
    
    #%%
    ####################################################################################
    ##                              SAVE DATA INTO .TXT
    ####################################################################################
    #Error mean
    ep_pf_mean = np.array(np.mean(np.matrix(ep_pf),axis=0))[0]
    ep_map_mean = np.array(np.mean(np.matrix(ep_map),axis=0))[0]
    ep_map2_mean = np.array(np.mean(np.matrix(ep_map2),axis=0))[0]
    ep_ml_mean = np.array(np.mean(np.matrix(ep_ml),axis=0))[0]
    ep_lsml_mean = np.array(np.mean(np.matrix(ep_lsml),axis=0))[0]
    ep_ls_mean = np.array(np.mean(np.matrix(ep_ls),axis=0))[0]
    #Error std
    ep_pf_std = np.array(np.std(np.matrix(ep_pf),axis=0))[0]
    ep_map_std = np.array(np.std(np.matrix(ep_map),axis=0))[0]
    ep_map2_std = np.array(np.std(np.matrix(ep_map2),axis=0))[0]
    ep_ml_std = np.array(np.std(np.matrix(ep_ml),axis=0))[0]
    ep_lsml_std = np.array(np.std(np.matrix(ep_lsml),axis=0))[0]
    ep_ls_std = np.array(np.std(np.matrix(ep_ls),axis=0))[0]
    #Process time mean
    t_pf_mean = np.array(np.mean(np.matrix(t_pf),axis=0))[0]
    t_map_mean = np.array(np.mean(np.matrix(t_map),axis=0))[0]
    t_map2_mean = np.array(np.mean(np.matrix(t_map2),axis=0))[0]
    t_ml_mean = np.array(np.mean(np.matrix(t_ml),axis=0))[0]
    t_lsml_mean = np.array(np.mean(np.matrix(t_lsml),axis=0))[0]
    t_ls_mean = np.array(np.mean(np.matrix(t_ls),axis=0))[0]
    #x scale
    t_offset = timet_aux[0]
    timet=np.array(timet_aux)-t_offset
       
    ep_pf_all = np.array(np.matrix(ep_pf).reshape(1,len(ep_pf)*len(ep_pf[0])))[0]
    ep_map_all = np.array(np.matrix(ep_map).reshape(1,len(ep_map)*len(ep_map[0])))[0]
    ep_map2_all = np.array(np.matrix(ep_map2).reshape(1,len(ep_map2)*len(ep_map2[0])))[0]
    ep_ml_all = np.array(np.matrix(ep_ml).reshape(1,len(ep_ml)*len(ep_ml[0])))[0]
    ep_ls_all = np.array(np.matrix(ep_ls).reshape(1,len(ep_ls)*len(ep_ls[0])))[0]
    ep_lsml_all = np.array(np.matrix(ep_lsml).reshape(1,len(ep_lsml)*len(ep_lsml[0])))[0]
    timet_all = np.array([])
    for i in range(len(ep_pf)):
        timet_all=np.concatenate((timet_all,timet),axis=0)
    #save the results obtained
    import os
    SAVE_FOLDER = 'Results11_it_it%d_sd%.4f_out%s'%(iteration_number,toa_noise_sd,toa_noise_outlier)
    if os.path.exists(SAVE_FOLDER) == False:
        os.mkdir(SAVE_FOLDER)
        
    aux_A = np.concatenate((timet,ep_pf_mean,ep_map_mean,ep_map2_mean,ep_ml_mean,ep_ls_mean,ep_lsml_mean,ep_pf_std,ep_map_std,ep_map2_std,ep_ml_std,ep_ls_std,ep_lsml_std,t_pf_mean,t_map_mean,t_map2_mean,t_ml_mean,t_ls_mean,t_lsml_mean),axis=0).reshape(19,timet.size)
    np.savetxt(SAVE_FOLDER+'/error_mean.txt', aux_A.T, delimiter=',',header='timet,ep_pf_mean,ep_map_mean,ep_map2_mean,ep_ml_mean,ep_ls_mean,ep_lsml_mean,ep_pf_std,ep_map_std,ep_map2_std,ep_ml_std,ep_ls_std,ep_lsml_std,t_pf_mean,t_map_mean,t_map2_mean,t_ml_mean,t_ls_mean,t_lsml_mean')
    aux_A = np.concatenate((timet_all,ep_pf_all,ep_map_all,ep_map2_all,ep_ml_all,ep_ls_all,ep_lsml_all),axis=0).reshape(7,timet_all.size)
    np.savetxt(SAVE_FOLDER+'/error_all.txt', aux_A.T, delimiter=',',header='timet_all,ep_pf_all,ep_map_all,ep_map2_all,ep_ml_all,ep_ls_all,ep_lsml_all')

    print 'PF    error = %.3f m (+/- %.3f m)'% (np.mean(ep_pf_mean), np.std(ep_pf_mean))
    print 'MAP   error = %.3f m (+/- %.3f m)'% (np.mean(ep_map_mean), np.std(ep_map_mean))
    print 'MAP_m error = %.3f m (+/- %.3f m)'% (np.mean(ep_map2_mean), np.std(ep_map2_mean))
    print 'LS    error = %.3f m (+/- %.3f m)'% (np.mean(ep_ls_mean), np.std(ep_ls_mean))
    print 'ML    error = %.3f m (+/- %.3f m)'% (np.mean(ep_ml_mean), np.std(ep_ml_mean))
    print 'LSML    error = %.3f m (+/- %.3f m)'% (np.mean(ep_lsml_mean), np.std(ep_lsml_mean))

#%%
####################################################################################
##                              PLOTS
####################################################################################
plt.figure(figsize=(9,3))
#plt.subplot(211)
plt.plot(timet,ep_map_mean,'m-',label='MAP')
plt.plot(timet,ep_map2_mean,'c-',label='MAP(Marg)')
plt.plot(timet,ep_pf_mean,'r-',label='PF')
plt.plot(timet,ep_ls_mean,'y-',label='LS')
plt.plot(timet,ep_ml_mean,'g-',label='ML')
plt.plot(timet,ep_lsml_mean,'b-',label='LSML')
plt.xlabel('Time (s)')
plt.ylabel('Position RMSE (m)')
pf_patch = mpatches.Patch(facecolor='r', label='$\overline{PF}$',linewidth=1,edgecolor='black')
map_patch = mpatches.Patch(facecolor='m', label='$\overline{MAP}$',linewidth=1,edgecolor='black')
map2_patch = mpatches.Patch(facecolor='c', label='$\overline{MAP_m}$',linewidth=1,edgecolor='black')
ls_patch = mpatches.Patch(facecolor='y', label='$\overline{LS}$',linewidth=1,edgecolor='black')
ml_patch = mpatches.Patch(facecolor='g', label='$\overline{ML}$',linewidth=1,edgecolor='black')
lsml_patch = mpatches.Patch(facecolor='b', label='$\overline{LSML}$',linewidth=1,edgecolor='black')
plt.legend(handles=[pf_patch,map_patch,map2_patch,ls_patch,ml_patch,lsml_patch],ncol=6,loc='upper right',fontsize=10)
plt.ylim(0,50)
plt.xlim(0,timet[-1])
plt.show()

plt.figure(figsize=(9,5))
plt.subplot(211)
plt.plot(timet,t_map_mean*1000,'m-',label='MAP')
plt.plot(timet,t_map2_mean*1000,'c-',label='MAP(Marg)')
plt.plot(timet,t_pf_mean*1000,'r-',label='PF')
plt.plot(timet,t_ls_mean*1000,'y-',label='LS')
plt.plot(timet,t_ml_mean*1000,'g-',label='ML')
plt.plot(timet,t_lsml_mean*1000,'b-',label='LSML')
plt.xlabel('Simulation Time (s)')
plt.ylabel('Runtime (ms)')
pf_patch = mpatches.Patch(facecolor='r', label='$\overline{PF}$',linewidth=1,edgecolor='black')
map_patch = mpatches.Patch(facecolor='m', label='$\overline{MAP}$',linewidth=1,edgecolor='black')
map2_patch = mpatches.Patch(facecolor='c', label='$\overline{MAP_m}$',linewidth=1,edgecolor='black')
ls_patch = mpatches.Patch(facecolor='y', label='$\overline{LS}$',linewidth=1,edgecolor='black')
ml_patch = mpatches.Patch(facecolor='g', label='$\overline{ML}$',linewidth=1,edgecolor='black')
lsml_patch = mpatches.Patch(facecolor='b', label='$\overline{LSML}$',linewidth=1,edgecolor='black')
plt.legend(handles=[pf_patch,map_patch,map2_patch,ls_patch,ml_patch,lsml_patch],ncol=6,loc='upper right',fontsize=10)
plt.yscale('log')
plt.show()

objects = ('$\overline{LS}$','$\overline{ML}$','$\overline{LSML}$','$\overline{MAP}$','$\overline{MAP}_m$','$\overline{PF}$')
y_pos = np.arange(len(objects))
performance = [np.mean(t_ls_mean[timet.size/2:])*1000.,np.mean(t_ml_mean[timet.size/2:])*1000.,np.mean(t_lsml_mean[timet.size/2:])*1000.,np.mean(t_map_mean[timet.size/2:])*1000.,np.mean(t_map2_mean[timet.size/2:])*1000.,np.mean(t_pf_mean[timet.size/2:])*1000.]
plt.figure(figsize=(5,5))
plt.bar(y_pos, performance, align='center', alpha=0.5,bottom=0.1,color='red')
plt.text(y_pos[0]-0.1,performance[0]+0.3,'%d'%(np.mean(t_ls_mean[timet.size/2:])*1000.))
plt.text(y_pos[1]-0.15,performance[1]+1.,'%d'%(np.mean(t_ml_mean[timet.size/2:])*1000.))
plt.text(y_pos[2]-0.15,performance[2]+1.,'%d'%(np.mean(t_lsml_mean[timet.size/2:])*1000.))
plt.text(y_pos[3]-0.2,performance[3]+80.5,'%d'%(np.mean(t_map_mean[timet.size/2:])*1000.))
plt.text(y_pos[4]-0.2,performance[4]+80.5,'%d'%(np.mean(t_map2_mean[timet.size/2:])*1000.))
plt.text(y_pos[5]-0.2,performance[5]+80.5,'%d'%(np.mean(t_pf_mean[timet.size/2:])*1000.))
plt.xticks(y_pos, objects)
plt.ylabel('Avg. runtime (ms)')
plt.yscale('log')
plt.ylim(0.1,10000)
plt.show()

plt.figure(figsize=(9,3))
plt.plot(timet,ep_pf_mean,'r-',label='PF')
plt.fill_between(timet,ep_pf_mean+ep_pf_std,ep_pf_mean-ep_pf_std, color = 'r', alpha = 0.1)
plt.plot(timet,ep_map_mean,'m-',label='MAP')
plt.fill_between(timet,ep_map_mean+ep_map_std,ep_map_mean-ep_map_std, color = 'm', alpha = 0.1)
plt.plot(timet,ep_map2_mean,'c-',label='MAPm')
plt.fill_between(timet,ep_map2_mean+ep_map2_std,ep_map2_mean-ep_map2_std, color = 'c', alpha = 0.1)
plt.plot(timet,ep_ls_mean,'y-',label='LS')
plt.fill_between(timet,ep_ls_mean+ep_ls_std,ep_ls_mean-ep_ls_std, color = 'y', alpha = 0.1)
plt.plot(timet,ep_ml_mean,'g-',label='ML')
plt.fill_between(timet,ep_ml_mean+ep_ml_std,ep_ml_mean-ep_ml_std, color = 'g', alpha = 0.1)
plt.plot(timet,ep_lsml_mean,'b-',label='LSML')
plt.fill_between(timet,ep_lsml_mean+ep_lsml_std,ep_lsml_mean-ep_lsml_std, color = 'b', alpha = 0.1)
plt.xlabel('Time (s)')
plt.ylabel('Position RMSE (m)')
#plt.title('TOA noise equal to %.3f s and 1 outlier'%toa_noise_sd)
plt.title('TOA noise equal to %.3f s'%toa_noise_sd)
pf_patch = mpatches.Patch(facecolor='r', label='$\overline{PF}$',linewidth=1,edgecolor='black')
map_patch = mpatches.Patch(facecolor='m', label='$\overline{MAP}$',linewidth=1,edgecolor='black')
map2_patch = mpatches.Patch(facecolor='c', label='$\overline{MAP_m}$',linewidth=1,edgecolor='black')
ls_patch = mpatches.Patch(facecolor='y', label='$\overline{LS}$',linewidth=1,edgecolor='black')
ml_patch = mpatches.Patch(facecolor='g', label='$\overline{ML}$',linewidth=1,edgecolor='black')
lsml_patch = mpatches.Patch(facecolor='b', label='$\overline{LSML}$',linewidth=1,edgecolor='black')
plt.legend(handles=[pf_patch,map_patch,map2_patch,ls_patch,ml_patch,lsml_patch],ncol=6,loc='upper right',fontsize=10)
plt.ylim(0,100)
plt.xlim(0,timet[-1])
plt.show()


print 'PF    error = %.3f m (+/- %.3f m)'% (np.mean(ep_pf_mean), np.std(ep_pf_mean))
print 'MAP   error = %.3f m (+/- %.3f m)'% (np.mean(ep_map_mean), np.std(ep_map_mean))
print 'MAP_m error = %.3f m (+/- %.3f m)'% (np.mean(ep_map2_mean), np.std(ep_map2_mean))
print 'LS    error = %.3f m (+/- %.3f m)'% (np.mean(ep_ls_mean), np.std(ep_ls_mean))
print 'ML    error = %.3f m (+/- %.3f m)'% (np.mean(ep_ml_mean), np.std(ep_ml_mean))
print 'LSML    error = %.3f m (+/- %.3f m)'% (np.mean(ep_lsml_mean), np.std(ep_lsml_mean))





