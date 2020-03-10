# -*- coding: utf-8 -*-
"""
Created on Fri March 13 14:10:49 2020

@author: Ivan Masmitja
"""


import matplotlib.pyplot as plt
import numpy as np
from numpy import cos
from numpy import sin
import matplotlib.patches as mpatches
from TargetClass_V11 import target_class
import os
########################################################################################
########################################################################################
printplot = True

######################################################################################################
# Functions
######################################################################################################

def visualization(A1,A2,A3,A4,step,t1,printplot):
    if printplot == True:
        plt.figure(figsize=(5,5))
        plt.title('Filter, step ' + str(step))

        plt.plot(A1[0],A1[1],'rx',ms=10,lw=1)
        plt.plot(A2[0],A2[1],'rx',ms=10,lw=1)
        plt.plot(A3[0],A3[1],'rx',ms=10,lw=1)
        plt.plot(A4[0],A4[1],'rx',ms=10,lw=1)

    particleprint = False 
    if particleprint == True:
        xx = t1.pf.x.T[0]
        yy = t1.pf.x.T[2]
        plt.plot(xx,yy,'bo',alpha=0.1,ms=1)


    if printplot == True:
        try:
            #print target position estimation
            arrow = plt.arrow(t1.pxs[-1][0],t1.pxs[-1][2],2*cos(t1.pxs[-1][4]),2*sin(t1.pxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='#fff0f5',edgecolor='#0000cc',alpha=0.4)
            plt.gca().add_patch(arrow) 
        except:
            next
            
        try:
            #print target position estimation using MAP
            arrow = plt.arrow(t1.mxs[-1][0],t1.mxs[-1][2],2*cos(t1.mxs[-1][4]),2*sin(t1.mxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='m',edgecolor='#0000cc',alpha=0.4)
            plt.gca().add_patch(arrow)
        except:
            next
            
        try:
             #print target position estimation using MAP
            arrow = plt.arrow(t1.mmxs[-1][0],t1.mmxs[-1][2],2*cos(t1.mmxs[-1][4]),2*sin(t1.mmxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='#90ee90',edgecolor='#0000cc',alpha=0.4)
            plt.gca().add_patch(arrow)
        except:
            next
            
        try:
            #print target position estimation using LS
            arrow = plt.arrow(t1.lsxs[-1][0],t1.lsxs[-1][2],2*cos(t1.lsxs[-1][4]),2*sin(t1.lsxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='yellow',edgecolor='#0000cc',alpha=0.4)
            plt.gca().add_patch(arrow)
        except:
            next
            
        try:
            #print target position estimation using ML
            arrow = plt.arrow(t1.mlxs[-1][0],t1.mlxs[-1][2],2*cos(t1.mlxs[-1][4]),2*sin(t1.mlxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='green',edgecolor='#0000cc',alpha=0.4)
            plt.gca().add_patch(arrow)
        except:
            next
            
        try:
            #print target position estimation using LSML
            arrow = plt.arrow(t1.lsmlxs[-1][0],t1.lsmlxs[-1][2],2*cos(t1.lsmlxs[-1][4]),2*sin(t1.lsmlxs[-1][4]),shape='full',lw=1,length_includes_head=False,head_width=40,facecolor='b',edgecolor='#0000cc',alpha=0.4)
            plt.gca().add_patch(arrow)
        except:
            next
            
        pf_patch = mpatches.Patch(facecolor='#fff0f5', label='$\overline{PF}$',linewidth=1,edgecolor='black')
        map_patch = mpatches.Patch(facecolor='m', label='$\overline{MAP}$',linewidth=1,edgecolor='black')
        map2_patch = mpatches.Patch(facecolor='#90ee90', label='$\overline{MAP}_m$',linewidth=1,edgecolor='black')
        ls_patch = mpatches.Patch(facecolor='yellow', label='$\overline{LS}$',linewidth=1,edgecolor='black')
        ml_patch = mpatches.Patch(facecolor='green', label='$\overline{ML}$',linewidth=1,edgecolor='black')
        lsml_patch = mpatches.Patch(facecolor='b', label='$\overline{LSML}$',linewidth=1,edgecolor='black')
        plt.legend(handles=[pf_patch,map_patch,map2_patch,ls_patch,ml_patch,lsml_patch],ncol=1,bbox_to_anchor=(1.4, 1.02))
        #Print when wg acquires a new range

        try:
            plt.plot(np.array(np.matrix(t1.mxs).T[0])[0],np.array(np.matrix(t1.mxs).T[2])[0],'m*--',alpha=0.3)
        except:
            next
        
        try:
            plt.plot(np.array(np.matrix(t1.mmxs).T[0])[0],np.array(np.matrix(t1.mmxs).T[2])[0],'c*--',color='#90ee90',alpha=0.3)
        except:
            next
        
        try:
            plt.plot(np.array(np.matrix(t1.lsxs).T[0])[0],np.array(np.matrix(t1.lsxs).T[2])[0],'y*--',alpha=0.3)
        except:
            next
        
        try:
            plt.plot(np.array(np.matrix(t1.mlxs).T[0])[0],np.array(np.matrix(t1.mlxs).T[2])[0],'g*--',alpha=0.3)
        except:
            next
        
        try:
            plt.plot(np.array(np.matrix(t1.lsmlxs).T[0])[0],np.array(np.matrix(t1.lsmlxs).T[2])[0],'b*--',alpha=0.3)
        except:
            next
        
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.axis('equal')
    plt.xlim(-400,400)
    plt.ylim(-400,400)

    plt.show()
    return()




#%%
###############################################################################
###                  Parameters initialization                             ####
###############################################################################

########################################################################################
#%%
###Acoustic receiver localizations
filename_BSP = 'TagDetectionsRESNEP7_BSP\position_BS_.txt'
easting, northing, depth = np.loadtxt(filename_BSP,skiprows=1, delimiter=',',usecols=(0,1,2), unpack=True)
A1 = np.array([easting.item(0),northing.item(0),depth.item(0)])
A2 = np.array([easting.item(1),northing.item(1),depth.item(1)])
A3 = np.array([easting.item(2),northing.item(2),depth.item(2)])
A4 = np.array([easting.item(3),northing.item(3),depth.item(3)])

#seting BS1 as (0,0)
bs_offset = A1[:-1] + np.array([50., -50.])
A1[:-1] -= bs_offset
A2[:-1] -= bs_offset
A3[:-1] -= bs_offset
A4[:-1] -= bs_offset

tag_id_all = ['A69-1601-60592','A69-1601-60593','A69-1601-65014','A69-1601-65015','A69-1602-14456','A69-1602-14457','A69-1602-14458','A69-1602-14459','A69-1602-14460','A69-1602-14461','A69-1602-14462','A69-1602-14463','A69-1602-14464','A69-1602-14465','A69-1602-14466','A69-1602-14467','A69-1602-14468','A69-1602-14469','A69-1602-14470','A69-1602-14471','A69-1602-14472','A69-1602-14473','A69-1602-14474','A69-1602-14475','A69-1602-14476','A69-1602-14477','A69-1602-14478','A69-1602-14479','A69-1602-14480','A69-1602-14481','A69-1602-14482','A69-1602-14483','A69-1602-14484','A69-1602-14485','A69-1602-14590','A69-1602-15084','A69-1602-15829','A69-1602-15830','A69-1602-15831','A69-1602-65266']
#tag_id_all = ['A69-1602-15829','A69-1602-15830','A69-1602-15831','A69-1602-65266']
num_tags = len(tag_id_all)
counter1 = 0
for tag_id in tag_id_all:
    counter1 += 1
    filename_BS1 = 'TagDetectionsRESNEPtest_BS1\detections_BS1_'+tag_id+'.txt'
    filename_BS2 = 'TagDetectionsRESNEPtest_BS2\detections_BS2_'+tag_id+'.txt'
    filename_BS3 = 'TagDetectionsRESNEPtest_BS3\detections_BS3_'+tag_id+'.txt'
    filename_BS4 = 'TagDetectionsRESNEPtest_BS4\detections_BS4_'+tag_id+'.txt'
    
    timestamps_BS1_d = np.loadtxt(filename_BS1,skiprows=1, delimiter=',',usecols=(0), unpack=True)
    timestamps_BS2_d = np.loadtxt(filename_BS2,skiprows=1, delimiter=',',usecols=(0), unpack=True)
    timestamps_BS3_d = np.loadtxt(filename_BS3,skiprows=1, delimiter=',',usecols=(0), unpack=True)
    timestamps_BS4_d = np.loadtxt(filename_BS4,skiprows=1, delimiter=',',usecols=(0), unpack=True)
    
    num_detections = timestamps_BS1_d.size
    
    #Starting the timestamp equalt to 0 in order to eliminate possible errors due to too big times
    t_offset = int(np.max(np.array([timestamps_BS1_d[0],timestamps_BS2_d[0],timestamps_BS3_d[0],timestamps_BS4_d[0]])))
    timestamps_BS1_d -= t_offset
    timestamps_BS2_d -= t_offset
    timestamps_BS3_d -= t_offset
    timestamps_BS4_d -= t_offset
    
    #set taget class
    t1=target_class(np.array([0.,0.,0.,0.]),np.array([0.,0.,0.,0.]))
    
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
    timestamp = []
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
    
    t_pf = []
    t_map = []
    t_map2 = []
    t_ml = []
    t_lsml = []
    t_ls = []
    dt = 0.01
        
    #Time buffer constants
    pft=[]
    ekft=[]
    ukft=[]
    mapt=[]
    map2t=[]
    lst=[]
    mlt=[]
    lsmlt=[]
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
            
        #Compute the tag position at new reception only if the tag has been received by at least 3 receivers
        RT = np.array(rt_aux)
        BS = np.array(BS_aux)
        if RT.size > 2:
            detection_number += 1
            ###############################################################################
            ###                        Compute positions                                ###
            #############################################################################
            # 2- Adquiere a new measurement
            #Set dt
            if old_rt == -1:
                old_rt = RT.item(0) - 10.01
            dt = RT.item(0) - old_rt
            old_rt = RT.item(0)
            timet_aux.append(RT.item(0)+t_offset)
            
            # 3- Compute the target estimation using different algorithms 
            lsml_time   = t1.updateLSMLE(dt,RT+0.,BS+0.)
                        
            #5- Visualitzation (Plot print)
            print 'Tag %s. Tag #%d of %d. Detections #%d of %d'%(tag_id, counter1, num_tags, detection_number, num_detections) 
            if printplot == True:
                visualization(A1,A2,A3,A4,detection_number,t1,printplot) #Function to print the plot and save variables
            lsmlt.append(lsml_time)
            ####################################################################################
            ##            
            ###################################################################################
    #Save position
    SAVE_FOLDER = 'positionV1'
    timestamp = np.array(timet_aux)
    try:
        targetpredictionx_pf = np.array(np.matrix(t1.pxs).T[0])[0] + bs_offset.item(0)
        targetpredictiony_pf = np.array(np.matrix(t1.pxs).T[2])[0] + bs_offset.item(1)
        folder = SAVE_FOLDER+'_PF'
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        aux_A = np.concatenate((timestamp,targetpredictionx_pf,targetpredictiony_pf),axis=0).reshape(3,timestamp.size)
        np.savetxt(folder+'/position_%s.txt'%tag_id, aux_A.T, delimiter=',',header='timestamp,easting,northing')
    except:
        next
    try:
        targetpredictionx_map = np.array(np.matrix(t1.mxs).T[0])[0] + bs_offset.item(0)
        targetpredictiony_map = np.array(np.matrix(t1.mxs).T[2])[0] + bs_offset.item(1)
        folder = SAVE_FOLDER+'_MAP'
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        aux_A = np.concatenate((timestamp,targetpredictionx_map,targetpredictiony_map),axis=0).reshape(3,timestamp.size)
        np.savetxt(folder+'/position_%s.txt'%tag_id, aux_A.T, delimiter=',',header='timestamp,easting,northing')
    except:
        next
    try:
        targetpredictionx_map2 = np.array(np.matrix(t1.mmxs).T[0])[0] + bs_offset.item(0)
        targetpredictiony_map2 = np.array(np.matrix(t1.mmxs).T[2])[0] + bs_offset.item(1)
        folder = SAVE_FOLDER+'_MAPm'
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        aux_A = np.concatenate((timestamp,targetpredictionx_map2,targetpredictiony_map2),axis=0).reshape(3,timestamp.size)
        np.savetxt(folder+'/position_%s.txt'%tag_id, aux_A.T, delimiter=',',header='timestamp,easting,northing')
    except:
        next
    try:
        targetpredictionx_ls = np.array(np.matrix(t1.lsxs).T[0])[0] + bs_offset.item(0)
        targetpredictiony_ls = np.array(np.matrix(t1.lsxs).T[2])[0] + bs_offset.item(1)
        folder = SAVE_FOLDER+'_LS'
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        aux_A = np.concatenate((timestamp,targetpredictionx_ls,targetpredictiony_ls),axis=0).reshape(3,timestamp.size)
        np.savetxt(folder+'/position_%s.txt'%tag_id, aux_A.T, delimiter=',',header='timestamp,easting,northing')
    except:
        next
    try:
        targetpredictionx_ml = np.array(np.matrix(t1.mlxs).T[0])[0] + bs_offset.item(0)
        targetpredictiony_ml = np.array(np.matrix(t1.mlxs).T[2])[0] + bs_offset.item(1)
        folder = SAVE_FOLDER+'_ML'
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        aux_A = np.concatenate((timestamp,targetpredictionx_ml,targetpredictiony_ml),axis=0).reshape(3,timestamp.size)
        np.savetxt(folder+'/position_%s.txt'%tag_id, aux_A.T, delimiter=',',header='timestamp,easting,northing')
    except:
        next
    try:    
        targetpredictionx_lsml = np.array(np.matrix(t1.lsmlxs).T[0])[0] + bs_offset.item(0)
        targetpredictiony_lsml = np.array(np.matrix(t1.lsmlxs).T[2])[0] + bs_offset.item(1)
        folder = SAVE_FOLDER+'_LSML'
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        aux_A = np.concatenate((timestamp,targetpredictionx_lsml,targetpredictiony_lsml),axis=0).reshape(3,timestamp.size)
        np.savetxt(folder+'/position_%s.txt'%tag_id, aux_A.T, delimiter=',',header=tag_id+': timestamp,easting,northing')
    except:
        next
    






