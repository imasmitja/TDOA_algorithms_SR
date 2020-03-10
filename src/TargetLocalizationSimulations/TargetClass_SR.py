# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:30:22 2017

Classes to define the target and its estimations using:
    1- EKF: Extended Kalman Filter
    2- UKF: Uncented Kalman Filter
    3- PF : Particle Filter
    4- MAP: Maximum A Posteriory estimation
    
@author: Ivan Masmitja
"""

import random
import numpy as np
from numpy import sqrt
from numpy import exp
from numpy import pi
from numpy import eye, zeros, dot, isscalar, outer
from filterpy.kalman import KalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from filterpy.common import dot3
from filterpy.kalman import unscented_transform
from filterpy.stats import logpdf
from filterpy.common import setter, setter_1d, setter_scalar
import scipy.linalg as linalg
from scipy.linalg import inv, cholesky
import time

import matplotlib.pyplot as plt
import matplotlib as mpl


#%%
####################################################################################################
##########                          EKF used in MAP                      ###########################
####################################################################################################
class ExtendedKalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        """ Extended Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        Parameters
        ----------

        dim_x : int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.

            This is used to set the default size of P, Q, and u

        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.
        """

        self.dim_x = dim_x
        self.dim_z = dim_z

        self._x = zeros((dim_x,1)) # state
        self._P = eye(dim_x)       # uncertainty covariance
        self._B = 0                # control transition matrix
        self._F = 0                # state transition matrix
        self._R = eye(dim_z)       # state uncertainty
        self._Q = eye(dim_x)       # process uncertainty
        self._y = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)


    def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):
        """ Performs the predict/update innovation of the extended Kalman
        filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.
            If `None`, only predict step is perfomed.

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, along with the
           optional arguments in args, and returns H.

        Hx : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.

        args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable.

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx after the required state
            variable.

        u : np.array or scalar
            optional control vector input to the filter.
        """

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        F = self._F
        B = self._B
        P = self._P
        Q = self._Q
        R = self._R
        x = self._x

        H = HJacobian(x, *args)

        # predict step
        x = dot(F, x) + dot(B, u)
        P = dot3(F, P, F.T) + Q

        # update step
        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))

        self._x = x + dot(K, (z - Hx(x, *hx_args)))

        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
               residual=np.subtract):
        """ Performs the update innovation of the extended Kalman filter.

        Parameters
        ----------

        z : np.array
            measurement for this step.
            If `None`, only predict step is perfomed.

        HJacobian : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.

        Hx : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable. for robot localization you might need to pass in
            information about the map and time of day, so you might have
            `args=(map_data, time)`, where the signature of HCacobian will
            be `def HJacobian(x, map, t)`

        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state
            variable.

        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        P = self._P
        if R is None:
            R = self._R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        x = self._x

        H = HJacobian(x, *args)

        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))

        hx =  Hx(x, *hx_args)
        
        y = residual(z, hx)
#        print 'predictedd range = '+str(hx)
#        print 'true       range = '+str(z)
#        print '        residual = '+str(y)
#        print 'Ky = '+str(dot(K,y))
        self._x = x + dot(K, y)

        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def predict_x(self, u=0):
        """ predicts the next state of X. If you need to
        compute the next state yourself, override this function. You would
        need to do this, for example, if the usual Taylor expansion to
        generate F is not providing accurate results for you. """
              
        self._x = dot(self._F, self._x) + dot(self._B, u)


    def predict(self, dt, u=0):
        """ Predict next position.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """
#        self._F = self.statetransition(dt)
        self.predict_x(u)
        self._P = dot3(self._F, self._P, self._F.T) + self._Q


    @property
    def Q(self):
        """ Process uncertainty matrix"""
        return self._Q


    @Q.setter
    def Q(self, value):
        """ Process uncertainty matrix"""
        self._Q = setter_scalar(value, self.dim_x)


    @property
    def P(self):
        """ state covariance matrix"""
        return self._P


    @P.setter
    def P(self, value):
        """ state covariance matrix"""
        self._P = setter_scalar(value, self.dim_x)


    @property
    def R(self):
        """ measurement uncertainty"""
        return self._R


    @R.setter
    def R(self, value):
        """ measurement uncertainty"""
        self._R = setter_scalar(value, self.dim_z)


    @property
    def F(self):
        """State Transition matrix"""
        return self._F


    @F.setter
    def F(self, value):
        """State Transition matrix"""
        self._F = setter(value, self.dim_x, self.dim_x)


    @property
    def B(self):
        """ control transition matrix"""
        return self._B


    @B.setter
    def B(self, value):
        """ control transition matrix"""
        self._B = setter(value, self.dim_x, self.dim_u)


    @property
    def x(self):
        """ state estimate vector """
        return self._x

    @x.setter
    def x(self, value):
        """ state estimate vector """
        self._x = setter_1d(value, self.dim_x)

    @property
    def K(self):
        """ Kalman gain """
        return self._K

    @property
    def y(self):
        """ measurement residual (innovation) """
        return self._y

    @property
    def S(self):
        """ system uncertainty in measurement space """
        return self._S

############################################################
#############################################################

#%%
#############################################################
## Particle Filter
############################################################
#For modeling the target we will use the TargetClass with the following attributes 
#and functions:
class ParticleFilter(object):
    """ Class for the Particle Filter """
 
    def __init__(self,std_range,init_velocity,dimx,particle_number):
 
        self.std_range = std_range
        self.init_velocity = init_velocity 
        self.x = zeros([particle_number,dimx])
        self.x_old = zeros([particle_number,dimx])
        self.particle_number = particle_number
        
        self.p_error = 0
        self._x = zeros([dimx]) 
       
        # target's noise
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0
        self.velocity_noise = 0.0
        
        # time interval
        self.dimx=dimx
        
        self._velocity = 0
        self._orientation = 0
        
        #Weights
        self.w = zeros(particle_number)
        
        #Flag to initialize the particles
        self.initialized = False
        

    def init_particles(self,position_A,position_B,tdoa):
        offset = np.array([0.,0.,0.,0.])
        p_A = position_A - offset
        p_B = position_B - offset
        
        for i in range(self.particle_number):           
            #Random distribution with hyperbola shape
            tdoa_variance = 0.005 #5ms
            tdoa += (np.random.rand()*tdoa_variance*2-tdoa_variance) #add noise to tdoa
            d = np.sqrt((p_B.item(0)-p_A.item(0))**2+(p_B.item(2)-p_A.item(2))**2)
            t = np.pi*np.random.rand()
            a = 1500*tdoa/2.
            if a == 0.:
                a += 1e-12
            e = 0.5*d/a
            if e<1.:
                e=1.
            b= np.sqrt(a**2*(e**2-1))
            sign = np.random.rand()
            if sign > 0.5:
                sign = -1
            else:
                sign = 1
            x_aux = a*np.cosh(t)+0.5*d
            y_aux = sign*b*np.sinh(t)
            
            #Rotation
            cc = (p_B.item(0)-p_A.item(0))
            co = (p_B.item(2)-p_A.item(2))
            if cc == 0:
                cc += 1e-12
            theta = np.arctan(co/cc)
            if cc<0:
                theta += np.pi
            self.x[i][0] = x_aux*np.cos(theta) - y_aux*np.sin(theta) + offset.item(0)
            self.x[i][2] = x_aux*np.sin(theta) + y_aux*np.cos(theta) + offset.item(2)
            
            #target's orientation
            orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
            # target's velocity 
            v = random.gauss(self.init_velocity, self.init_velocity/2)  
            self.x[i][1] = np.cos(orientation)*v
            self.x[i][3] = np.sin(orientation)*v
       
        self.initialized = True
        return       
        
    #Noise parameters can be set by:
    def set_noise(self, forward_noise, turn_noise, sense_noise, velocity_noise):
        """ Set the noise parameters, changing them is often useful in particle filters
        :param new_forward_noise: new noise value for the forward movement
        :param new_turn_noise:    new noise value for the turn
        :param new_sense_noise:  new noise value for the sensing
        """
     
        #target's noise
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise
        self.velocity_noise = velocity_noise
        return

    #Move particles acording to its motion
    def predict(self,dt):
        """ Perform target's turn and move
        :param turn:    turn command
        :param forward: forward command
        :return target's state after the move
        """

        gaussnoise = False
        for i in range(self.particle_number):
            # turn, and add randomness to the turning command
            turn = np.arctan2(self.x[i][3],self.x[i][1])
            if gaussnoise == True:
                orientation = turn + random.gauss(0.0, self.turn_noise)
            else:
                orientation = turn +  np.random.rand()*self.turn_noise*2 -self.turn_noise
            orientation %= 2 * np.pi
         
            # move, and add randomness to the motion command
            velocity = np.sqrt(self.x[i][1]**2+self.x[i][3]**2)
            forward = velocity*dt
            if gaussnoise == True:
                dist = float(forward) + random.gauss(0.0, self.forward_noise)
            else:
                dist = float(forward) + np.random.rand()*self.forward_noise*2 - self.forward_noise
            self.x[i][0] = self.x[i][0] + (np.cos(orientation) * dist)
            self.x[i][2] = self.x[i][2] + (np.sin(orientation) * dist)
            if gaussnoise == True:
                newvelocity = velocity + random.gauss(0.0, self.velocity_noise)
            else:
                newvelocity = velocity + np.random.rand()*self.velocity_noise*2 - self.velocity_noise
            if newvelocity < 0:
                newvelocity = 0
            self.x[i][1] = np.cos(orientation) * newvelocity
            self.x[i][3] = np.sin(orientation) * newvelocity
     
        return 

    #To calculate Gaussian probability:
    @staticmethod
    def gaussian(mu, sigma, x):
        """ calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        :param mu:    distance to the landmark
        :param sigma: standard deviation
        :param x:     distance to the landmark measured by the target
        :return gaussian value
        """
     
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
    
    #The next function we will need to assign a weight to each particle according to 
    #the current measurement. See the text below for more details. It uses effectively a 
    #Gaussian that measures how far away the predicted measurements would be from the 
    #actual measurements. Note that for this function you should take care of measurement 
    #noise to prevent division by zero. Such checks are skipped here to keep the code 
    #as short and compact as possible.
    def measurement_prob(self, measurement,position_A,position_B):
        """ Calculate the measurement probability: how likely a measurement should be
        :param measurement: current measurement
        :return probability
        """
        #The closer a particle to a correct position, the more likely will be the set of 
        #measurements given this position. The mismatch of the actual measurement and the 
        #predicted measurement leads to a so-called importance weight. It tells us how important 
        #that specific particle is. The larger the weight, the more important it is. According 
        #to this each of our particles in the list will have a different weight depending on 
        #a specific target’s measurement. Some will look very plausible, others might look 
        #very implausible.
        # 4- generate particle weights depending on target's measurement

        for i in range(self.particle_number):
            d_tdoa = np.sqrt((self.x[i][0] - position_A[0])**2 + (self.x[i][2] - position_A[2])**2) - np.sqrt((self.x[i][0] - position_B[0])**2 + (self.x[i][2] - position_B[2])**2)
            v_tdoa = d_tdoa/1500.
            self.w[i] = self.gaussian(v_tdoa, self.sense_noise, measurement)
        return 
    
    def resampling(self):
        #After that we let these particles survive randomly, but the probability of survival 
        #will be proportional to the weights.
        #The final step of the particle filter algorithm consists in sampling particles from 
        #the list p with a probability which is proportional to its corresponding w value. 
        #Particles in p having a large weight in w should be drawn more frequently than the 
        #ones with a small value
        #Here is a pseudo-code of the resampling step:
        #while w[index] < beta:
        #    beta = beta - w[index]
        #    index = index + 1
        #    select p[index]
        self.x_old = self.x+0.
        method = 2
        if method == 1:   
            # 4- resampling with a sample probability proportional
            # to the importance weight
            p3 = zeros([self.particle_number,self.dimx])
            index = int(random.random() * self.particle_number)
            beta = 0.0
            mw = max(self.w)
            for i in range(self.particle_number):
                beta += random.random() * 2.0 * mw
                while beta > self.w[index]:
                    beta -= self.w[index]
                    index = (index + 1) % self.particle_number
                p3[i]=self.x[index]
            self.x = p3
            return
        if method == 2:
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            # Systematic Resampling
            p3 = zeros([self.particle_number,self.dimx])
            ci = zeros(self.particle_number)
            normalized_w = self.w/np.sum(self.w)
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = random.random()/self.particle_number
            i = 0
            for j in range(self.particle_number):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                u = u + 1./self.particle_number
            self.x = p3
            return
        if method == 3: #this mehtod works ok and was presented in OCEANS Kobe 2018
            # Systematic Resampling + random resampling
            if self.particle_number == 10000:
                ratio = 640 #160 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 6000:
                ratio = 400 #100 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 3000:
                ratio = 200 #50 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 1000:
                ratio = 120 #15 works ok; ratio=10 is ok for statik targets
            else:
                ratio = 50 #50 works ok; ratio=10 is ok for statik targets
            radii = 100 #50 works ok
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            p3 = zeros([self.particle_number,self.dimx])
            ci = zeros(self.particle_number)
            try:
                normalized_w = self.w/np.sum(self.w)
            except:
                normalized_w = self.w/(np.sum(self.w)+1e-12)
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = random.random()/(self.particle_number-ratio)
            i = 0
            for j in range((self.particle_number-ratio)):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                u = u + 1./(self.particle_number-ratio)
                
            for i in range(ratio):
                #Random distribution with circle shape
                aux=np.zeros(4)
                t = 2*np.pi*np.random.rand()
                r = np.random.rand()*radii
                aux[0] = r*np.cos(t)+self._x[0]
                aux[2] = r*np.sin(t)+self._x[2]
                #target's orientation
                orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
                # target's velocity 
                v = random.gauss(self.init_velocity, self.init_velocity/2.)  
                aux[1] = np.cos(orientation)*v
                aux[3] = np.sin(orientation)*v
                p3[j+i+1]= aux
                self.w[j+i+1] = 1./(self.particle_number/3.)
            self.x = p3
            return
        
        if method == 3.1: #this mehtod works ok and was presented in OCEANS Kobe 2018 but with resizing particle number            
            previous_p_number = self.particle_number
            previous_ratio = 50
#            if p_std >= 10:
            if self.p_error >= 2:
                self.particle_number = 10000
            else:
                self.particle_number = 3000
            # Systematic Resampling + random resampling
            if self.particle_number == 10000:
                ratio = 160 #50 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 6000:
                ratio = 100 #50 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 3000:
                ratio = 50 #50 works ok; ratio=10 is ok for statik targets
            elif self.particle_number == 1000:
                ratio = 15 #50 works ok; ratio=10 is ok for statik targets
            else:
                ratio = 50 #50 works ok; ratio=10 is ok for statik targets
            radii = 50 #50 works ok
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            p3 = zeros([self.particle_number,self.dimx])
            p3_w = zeros(self.particle_number)
            ci = zeros(previous_p_number)
            normalized_w = self.w/np.sum(self.w)
            ci[0]=normalized_w[0]
            for i in range(1,previous_p_number):
                ci[i]=ci[i-1]+normalized_w[i]
            u = random.random()/(previous_p_number-previous_ratio)
            i = 0
            for j in range((self.particle_number-ratio)):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                p3_w[j]=self.w[i]
                u = u + 1./(previous_p_number-previous_ratio)
                if i>= previous_p_number-previous_ratio:
                    i = 0
                    u = random.random()/(previous_p_number-previous_ratio)
#            print 'j=',j
                
            for i in range(ratio):
                #Random distribution with circle shape
                aux=np.zeros(4)
                t = 2*np.pi*np.random.rand()
                r = np.random.rand()*radii
                aux[0] = r*np.cos(t)+self._x[0]
                aux[2] = r*np.sin(t)+self._x[2]
                #target's orientation
                orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
                # target's velocity 
                v = random.gauss(self.init_velocity, self.init_velocity/2.)  
        #            v = np.random.rand() * self.init_velocity 
        #            v = self.init_velocity 
                aux[1] = np.cos(orientation)*v
                aux[3] = np.sin(orientation)*v
                p3[j+i+1]= aux
                p3_w[j+i+1] = 1./(self.particle_number/3.)
            self.x = p3
            self.w = p3_w
            previous_ratio = ratio
            return
                
        if method == 4:
            # Systematic Resampling + random resampling
            ratio = 50
            radii = 100
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            p3 = zeros([self.particle_number,self.dimx])
            ci = zeros(self.particle_number)
            normalized_w = self.w/np.sum(self.w)
#            print 'Normalized weights beffore= ', normalized_w
            for i in range(ratio):
                normalized_w=np.delete(normalized_w,np.argmin(normalized_w))
            normalized_w = normalized_w/np.sum(normalized_w)
#            print 'Normalized weights after= ', normalized_w
            ci[0]=normalized_w[0]
            for i in range(1,self.particle_number-ratio):
                ci[i]=ci[i-1]+normalized_w[i]
            u = random.random()/(self.particle_number-ratio)
            i = 0
            for j in range((self.particle_number-ratio)):
                while (u > ci[i]):
                    i += 1
                p3[j]=self.x[i]
                u = u + 1./(self.particle_number-ratio)
                
            for i in range(ratio):
                #Random distribution with circle shape
                aux=np.zeros(4)
                t = 2*np.pi*np.random.rand()
                r = np.random.rand()*radii
                aux[0] = r*np.cos(t)+self._x[0]
                aux[2] = r*np.sin(t)+self._x[2]
                #target's orientation
                orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
                # target's velocity 
                v = random.gauss(self.init_velocity, self.init_velocity/2.)  
                aux[1] = np.cos(orientation)*v
                aux[3] = np.sin(orientation)*v
                p3[j+i]= aux
                self.w[j+i] = 1/10.
            self.x = p3
            return        
        
        if method == 5: #it doesn't work
            # Systematic Resampling + random resampling
            ratio = 50
            radii = 100
            #From: https://classroom.udacity.com/courses/ud810/lessons/3353208568/concepts/33538586070923
            p3 = zeros([self.particle_number,self.dimx])
            ci = zeros(self.particle_number)
            normalized_w = self.w/np.sum(self.w)
            upper_threshold = max(normalized_w)*0.1
            lower_threshold = min(normalized_w)*10.
            i = 0
            ci = []
            pi = []
            wi = []
            for n in range (self.particle_number):
                if normalized_w[n] > upper_threshold:
                    p3[i] = self.x[n]
                    i += 1
                elif normalized_w[n] <= upper_threshold and normalized_w[n] >= lower_threshold:
                    pi.append(self.x[n])
                    wi.append(self.w[n])
                elif normalized_w[n] < lower_threshold:
                    #Random distribution with circle shape
                    aux=np.zeros(4)
                    t = 2*np.pi*np.random.rand()
                    r = np.random.rand()*radii
                    aux[0] = r*np.cos(t)+self._x[0]
                    aux[2] = r*np.sin(t)+self._x[2]
                    #target's orientation
                    orientation = np.random.rand() * 2.0 * np.pi   # target's orientation
                    # target's velocity 
                    v = random.gauss(self.init_velocity, self.init_velocity/2.)  
                    aux[1] = np.cos(orientation)*v
                    aux[3] = np.sin(orientation)*v
                    p3[i]= aux
                    self.w[n] = 1/1000.
                    i += 1
            normalized_wi = wi/np.sum(wi)
            ci = zeros(len(pi))
            ci[0]=normalized_wi[0]
            for i in range(1,len(pi)):
                ci[i]=ci[i-1]+normalized_wi[i]
            u = random.random()/(len(ci))
            i = 0
            for j in range(len(ci)):
                while (u > ci[i]):
                    i += 1
                p3[j+i]=pi[i]
                u = u + 1./(len(ci))
                              
            self.x = p3
            return        
    
    def target_estimation(self,position_A,position_B):
        """ Calculate the mean error of the system
        :param r: current target object
        :param p: particle set
        :return mean error of the system
        """
        #8- Target prediction (we predict the best estimation for target's position=mean of all particles)
        sumx = 0.0
        sumy = 0.0
        sumvx = 0.0
        sumvy = 0.0
        plotfig = False
        method = 2
#        method = 3 #Using k-mean clustering method
#        method = 4 #Using mean-shift clustering to find the center of all particles
#        method = 5 #Both 3 and 4. where 3 is used to initialize 4
        if method == 1:
            for i in range(self.particle_number):
               sumx += self.x[i][0]
               sumy += self.x[i][2]
               sumvx += self.x[i][1]
               sumvy += self.x[i][3]
            self._x = np.array([sumx, sumvx, sumy, sumvy])/self.particle_number
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
            
        if method == 2:
            for i in range(self.particle_number):
               sumx += self.x[i][0]*self.w[i]
               sumy += self.x[i][2]*self.w[i]
               sumvx += self.x[i][1]*self.w[i]
               sumvy += self.x[i][3]*self.w[i]    
            self._x = np.array([sumx, sumvx, sumy, sumvy])/np.sum(self.w)
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
        
        if method == 3:
            for i in range(self.particle_number):
               sumx += self.x[i][0]
               sumy += self.x[i][2]
               sumvx += self.x[i][1]
               sumvy += self.x[i][3]
            method1 = np.array([sumx, sumvx, sumy, sumvy])/self.particle_number
            sumx=0
            sumy=0
            sumvx=0
            sumvy=0
            for i in range(self.particle_number):
               sumx += self.x[i][0]*self.w[i]
               sumy += self.x[i][2]*self.w[i]
               sumvx += self.x[i][1]*self.w[i]
               sumvy += self.x[i][3]*self.w[i]    
            self._x = np.array([sumx, sumvx, sumy, sumvy])/np.sum(self.w)
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
            
            #Obtained from Standford Univeristi onlyn course
            #https://www.youtube.com/watch?v=PpH_hv55GNQ
            K = 5 #Number of clusters (Can be random)
            #iterations = 10 #number of itaration to do to avoid local minimum, and therefore, obtan the optimal solution
            #1- Random initialize centoids Step
            muc = [] #centroids position
            for i in range(K):
                random_particle = int(np.random.rand()*self.particle_number)
                muc.append([self.x[random_particle][0],self.x[random_particle][2],self.w[random_particle]])
            old_mu = list(muc)
            while(1):
                #2- Cluster assignament Step
                #initialize cluster
                cluster = []
                for i in range(K):
                    cluster.append([muc[i]])
                #fill cluster with all points
                for i in range(self.particle_number):
                    dist = []
                    for k in range(K):
                        distance = np.sqrt((self.x[i][0]-muc[k][0])**2+(self.x[i][2]-muc[k][1])**2)
                        dist.append(distance)
                    cluster[np.argmin(dist)].append([self.x[i][0],self.x[i][2],self.w[i]])             
                #3- Move centroids Step
                for i in range(K):
                    muc[i] = list(np.mean(np.array(cluster[i]),axis=0))
                variation = np.sum(np.array(old_mu)-np.array(muc))
                old_mu = list(muc)
                #4- Compute centroids weight
                cw = []
                for i in range(K):
                    cw.append(np.sum(np.array(cluster[i]).T[2]))
                if abs(variation)<5:
                    numtotake = 3
                    indi = np.argpartition(cw,-numtotake)[-numtotake:]
                    xxx = np.mean(np.array(muc)[indi].T[0])
                    yyy = np.mean(np.array(muc)[indi].T[1])
                    center_interes = [xxx,yyy]
                    if plotfig==True:
                        fig, ax = plt.subplots(figsize=(5,5))
                        plt.title('Particle Filter (PF): Step #')
                        for i in range(K):
                            plt.plot(np.array(cluster[i]).T[0],np.array(cluster[i]).T[1],marker='o',ms=5,alpha=0.2,lw=0)
                            plt.plot(muc[i][0],muc[i][1],'kx',ms=20,lw=20)
                            plt.annotate(str(i),xy=(muc[i][0], muc[i][1]))
                        plt.plot(self._x[0],self._x[2],'k^')
                        plt.plot(method1[0],method1[2],'k*')
                        plt.plot(center_interes[0],center_interes[1],'rx',ms=20,lw=20)
                        plt.xlabel('Easting [m]')
                        plt.ylabel('Northing [m]')                   
                        plt.ticklabel_format(useOffset=False)
                        plt.axis('equal')              
                        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(100.))
                        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(100.))
                        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(400.))
                        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(400.))
                        ax.grid(which = 'minor',axis='both', linestyle='-', linewidth=0.4)                                      
                        plt.show()
                        plt.clf()
                        plt.close()
                        print 'SUM:  centroids=',cw
                        print 'MEAN: Wcentroid1=%.3f, Wcentroid1=%.3f, Wcentroid1=%.3f'%(np.mean(np.array(cluster[0]).T[2]),np.mean(np.array(cluster[1]).T[2]),np.mean(np.array(cluster[2]).T[2]))
                    self._x = np.array([center_interes[0], self._x[1], center_interes[1], self._x[3]])
                    self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
                    self._orientation = np.arctan2(self._x[3],self._x[1])
                    break
                
        if method == 4:
            #obtained from 
            #https://classroom.udacity.com/courses/ud810/lessons/3587388584/concepts/34921993280923
            for i in range(self.particle_number):
               sumx += self.x[i][0]
               sumy += self.x[i][2]
               sumvx += self.x[i][1]
               sumvy += self.x[i][3]
            method1 = np.array([sumx, sumvx, sumy, sumvy])/self.particle_number
            sumx=0
            sumy=0
            sumvx=0
            sumvy=0
            
            for i in range(self.particle_number):
               sumx += self.x[i][0]*self.w[i]
               sumy += self.x[i][2]*self.w[i]
               sumvx += self.x[i][1]*self.w[i]
               sumvy += self.x[i][3]*self.w[i]    
            self._x = np.array([sumx, sumvx, sumy, sumvy])/np.sum(self.w)
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
            
            #1-Randomli take one particle as a center of interes
            #random_particle = int(np.random.rand()*self.particle_number)
            #center_interes = [self.x[random_particle][0],self.x[random_particle][2]]
            center_interes = [self._x[0],self._x[2]]
            while (True):
                #2-Compute the center of mas incide the area
                max_radius = 100 #maximum radius of the area of interest
                sumx=0.
                sumy=0.
                sumvx=0.
                sumvy=0.
                sumw=0.
                for i in range(self.particle_number):
                    dist_p = np.sqrt((self.x[i][0]-center_interes[0])**2+(self.x[i][2]-center_interes[1])**2)
                    if dist_p <= max_radius:
                        sumx += self.x[i][0]*self.w[i]
                        sumy += self.x[i][2]*self.w[i]
                        sumvx += self.x[i][1]*self.w[i]
                        sumvy += self.x[i][3]*self.w[i]
                        sumw += self.w[i]
                if sumw == 0:
                    sumw = 0.00001
                center_mass = np.array([sumx, sumvx, sumy, sumvy])/sumw
                #3- compute the mean shift vector
                msv = np.sqrt((center_interes[0]-center_mass[0])**2+(center_interes[1]-center_mass[2])**2)
                if plotfig==True:
                    fig, ax = plt.subplots(figsize=(5,5))
                    plt.title('Particle Filter (PF): Step #')
                    plt.plot(self.x.T[0],self.x.T[2],'bo',alpha=0.2)
                    plt.plot(center_mass[0],center_mass[2],'kx',ms=20,lw=20)
                    plt.plot(center_interes[0],center_interes[1],'ro')
                    circle2 = plt.Circle((center_interes[0], center_interes[1]), max_radius, color='r', fill=False)
                    ax.add_artist(circle2)
                    plt.plot(self._x[0],self._x[2],'k^')
                    plt.plot(method1[0],method1[2],'k*')
                    plt.xlabel('Easting [m]')
                    plt.ylabel('Northing [m]')                   
                    plt.ticklabel_format(useOffset=False)
                    plt.axis('equal')              
                    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(100.))
                    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(100.))
                    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(400.))
                    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(400.))
                    ax.grid(which = 'minor',axis='both', linestyle='-', linewidth=0.4)                                      
                    plt.show()
                    plt.clf()
                    plt.close()
                    print 'Mean Shift Vector = ',msv
                if msv <= 5:
                    self._x = np.array([sumx, sumvx, sumy, sumvy])/sumw
                    self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
                    self._orientation = np.arctan2(self._x[3],self._x[1])
                    break
                else:
                    center_interes = [center_mass[0],center_mass[2]]
            
        if method == 5:
            polar = True
            if plotfig==True:
                for i in range(self.particle_number):
                   sumx += self.x[i][0]
                   sumy += self.x[i][2]
                   sumvx += self.x[i][1]
                   sumvy += self.x[i][3]
                method1 = np.array([sumx, sumvx, sumy, sumvy])/self.particle_number
            
            sumx=0
            sumy=0
            sumvx=0
            sumvy=0
            for i in range(self.particle_number):
               sumx += self.x[i][0]*self.w[i]
               sumy += self.x[i][2]*self.w[i]
               sumvx += self.x[i][1]*self.w[i]
               sumvy += self.x[i][3]*self.w[i]    
            self._x = np.array([sumx, sumvx, sumy, sumvy])/np.sum(self.w)
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
            #Obtained from Standford Univeristi onlyn course
            #https://www.youtube.com/watch?v=PpH_hv55GNQ
            K = 5 #Number of clusters (Can be random)
            #iterations = 10 #number of itaration to do to avoid local minimum, and therefore, obtan the optimal solution
            #0- Represent all particles in a polar form where the center is the previous estimation self._x
            if polar==True:
                p_polar = []
                xp = self._x[0]
                yp = self._x[2]
                for i in range(self.particle_number):
                    vec = np.sqrt((self.x[i][0]-xp)**2+(self.x[i][2]-yp)**2)
                    ang = np.arctan2(self.x[i][2]-yp,self.x[i][0]-xp)
                    if ang<0:
                        ang+=2*np.pi
                    p_polar.append([vec,ang])
                maxvec=np.max(np.array(p_polar).T[0])
                maxang=np.max(np.array(p_polar).T[1])
                for i in range(self.particle_number):
                    p_polar[i][0]=p_polar[i][0]/maxvec
                    p_polar[i][1]=p_polar[i][1]/maxang
            numruns = 3
            allcenter_interes = []
            allj_range = []
            for runs in range(numruns):
                #1- Random initialize centoids Step
                muc = [] #centroids position
                for i in range(K):
                    random_particle = int(np.random.rand()*self.particle_number)
                    if polar == True:
                        muc.append([p_polar[random_particle][0],p_polar[random_particle][1],self.w[random_particle]])
                    else:
                        muc.append([self.x[random_particle][0],self.x[random_particle][2],self.w[random_particle]])
                old_mu = list(muc)
                while(1):
                    #2- Cluster assignament Step
                    #initialize cluster
                    cluster = []
                    for i in range(K):
                        cluster.append([muc[i]])
                    #fill cluster with all points
                    for i in range(self.particle_number):
                        dist = []
                        if polar == True:
                            for k in range(K):
                                distance = np.sqrt((p_polar[i][0]-muc[k][0])**2+(p_polar[i][1]-muc[k][1])**2)
                                dist.append(distance)
                            cluster[np.argmin(dist)].append([p_polar[i][0],p_polar[i][1],self.w[i]])       
                        else:
                            for k in range(K):
                                distance = np.sqrt((self.x[i][0]-muc[k][0])**2+(self.x[i][2]-muc[k][1])**2)
                                dist.append(distance)
                            cluster[np.argmin(dist)].append([self.x[i][0],self.x[i][2],self.w[i]])  
                    #3- Move centroids Step
                    for i in range(K):
                        muc[i] = list(np.mean(np.array(cluster[i]),axis=0))
                    variation = np.sum(np.array(old_mu)-np.array(muc))
                    old_mu = list(muc)
                    #4- Compute centroids weight
                    cw = []
                    for i in range(K):
                        cw.append(np.sum(np.array(cluster[i]).T[2]))
                    if polar == True:
                        variation_limit = 0.1
                    else:
                        variation_limit = 5
                    if abs(variation)<variation_limit:
                        j_range = []
                        for i in range(K):
                            try:
                                j_range.append(np.sqrt((np.std(np.array(cluster[i]).T[0]-muc[i][0]))**2+(np.std(np.array(cluster[i]).T[1]-muc[i][1]))**2))
                            except:
                                j_range.append(101)
                        allj_range.append(np.sum(j_range))
                        plottot2 = False
                        if plottot2 == True:
                            fig, ax = plt.subplots(figsize=(5,5))
                            for i in range(K):
                                plt.plot(np.array(cluster[i]).T[0],np.array(cluster[i]).T[1],marker='o',ms=2,alpha=0.2,lw=0)
                                plt.plot(muc[i][0],muc[i][1],'kx',ms=20,lw=20)
                            plt.show()
                        #when k-mean is finished we transform all the polar centroids into the x-y plane
                        if polar == True:
                            for i in range(K):
                                muc[i] = [np.cos(muc[i][1]*maxang)*muc[i][0]*maxvec+xp,np.sin(muc[i][1]*maxang)*muc[i][0]*maxvec+yp,muc[i][2]]
                                
                        allcenter_interes.append([muc[np.argmax(cw)][0],muc[np.argmax(cw)][1]])
                        break
            if True:
                #Runmean-shift algorithm over the main cluster
                #1-Randomli take one particle as a center of interes
                center_interes = allcenter_interes[np.argmin(allj_range)]
                while (True):
                    #2-Compute the center of mas incide the area
                    max_radius = 25 #maximum radius of the area of interest
                    sumx=0.
                    sumy=0.
                    sumvx=0.
                    sumvy=0.
                    sumw=0.
                    for i in range(self.particle_number):
                        dist_p = np.sqrt((self.x[i][0]-center_interes[0])**2+(self.x[i][2]-center_interes[1])**2)
                        if dist_p <= max_radius:
                            sumx += self.x[i][0]*self.w[i]
                            sumy += self.x[i][2]*self.w[i]
                            sumvx += self.x[i][1]*self.w[i]
                            sumvy += self.x[i][3]*self.w[i]
                            sumw += self.w[i]
                    center_mass = np.array([sumx, sumvx, sumy, sumvy])/sumw
                    #3- compute the mean shift vector
                    msv = np.sqrt((center_interes[0]-center_mass[0])**2+(center_interes[1]-center_mass[2])**2)                    
                    if msv <= 5:
                        break
                    else:
                        center_interes = [center_mass[0],center_mass[2]]
                if plotfig==True:
                    fig, ax = plt.subplots(figsize=(5,5))
                    plt.title('Particle Filter (PF): Step #')
                    plt.plot(self.x.T[0],self.x.T[2],'bo',alpha=0.2,ms=1)
                    for i in range(K):
#                        plt.plot(np.array(cluster[i]).T[0],np.array(cluster[i]).T[1],marker='o',ms=5,alpha=0.2,lw=0)
                        plt.plot(muc[i][0],muc[i][1],'kx',ms=20,lw=20)
                        plt.annotate(str(i),xy=(muc[i][0], muc[i][1]))
                    plt.plot(self._x[0],self._x[2],'k^')
                    plt.plot(method1[0],method1[2],'k*')
                    plt.plot(center_mass[0],center_mass[2],'rx',ms=20,lw=80)
                    plt.plot(center_interes[0],center_interes[1],'rx')
                    circle2 = plt.Circle((center_interes[0], center_interes[1]), max_radius, color='r', fill=False)
                    ax.add_artist(circle2)
                    plt.xlabel('Easting [m]')
                    plt.ylabel('Northing [m]')                   
                    plt.ticklabel_format(useOffset=False)
                    plt.axis('equal')              
                    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(100.))
                    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(100.))
                    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(400.))
                    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(400.))
                    ax.grid(which = 'minor',axis='both', linestyle='-', linewidth=0.4)                                      
                    plt.show()
                    plt.clf()
                    plt.close()
                    print 'SUM:  centroids=',cw                    
            self._x = center_mass+0.
            self._velocity = np.sqrt(self._x[1]**2+self._x[3]**2)
            self._orientation = np.arctan2(self._x[3],self._x[1])
        #finally the covariance matrix is computed. 
        #http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
        xarray = self.x.T[0]
        yarray = self.x.T[2]
        self.cov_matrix = np.cov(xarray, yarray)
        return
    
    #6- It computes the average error of each particle relative to the target pose. We call 
    #this function at the end of each iteration:
    # here we get a set of co-located particles   
    #At every iteration we want to see the overall quality of the solution, for this 
    #we will use the following function:
    def evaluation(self,position_A,position_B,z,max_error=50):
        """ Calculate the mean error of the system
        :param r: current target object
        :param p: particle set
        :return mean error of the system
        """
        sum2 = 0.0
        for i in range(self.particle_number):
            # Calculate the mean error of the system between Landmark (WG) and particle set
            dxa = (self.x[i][0] - position_A[0])
            dya = (self.x[i][2] - position_A[2])
            dxb = (self.x[i][0] - position_B[0])
            dyb = (self.x[i][2] - position_B[2])
            err = sqrt(dxa**2 + dya**2)-sqrt(dxb**2 + dyb**2)
            sum2 += err
        self.p_error = abs(sum2/self.particle_number - z)            
        if self.p_error > max_error:
            self.initialized = False
        return 
        
    
#%%
###########################################################################################################
##########################################################################################################
##############################                    TARGET CLASS   ##########################################
###########################################################################################################
##########################################################################################################

class target_class(object):
    
    def __init__(self,mytarget,myobserver):
        #Target parameters
        self.txs = []
        self.mytarget = mytarget +0.
        
        #Initial target prediction 
        mytargetp = np.array([myobserver[0]+1, 0., myobserver[2], 0.])
        
        #Parameters of simulation
        dt = 60. #time in secons equivalent for each iteration t
        self.allz=[]
        
        #PF initialization###############################################################################################
        self.pxs=[]
        self.pf_dt = 0
        #Our particle filter will maintain a set of n random guesses (particles) where 
        #the target might be. Each guess (or particle) is a vector containing [x,vx,y,vy]
        # create a set of particles
        self.pf = ParticleFilter(std_range=0.01,init_velocity=0.5,dimx=4,particle_number=6000)
        self.pf.set_noise(forward_noise = 5., turn_noise = 6., sense_noise=0.005, velocity_noise = 0.1) #turn noise: 0.1=5degrees 3.=180degrees
        
        #MAP initialization#########################################################################################
        self.mxs=[]
        self.m_dt = 0
        self.m_rtsize = []
        self.mape = ExtendedKalmanFilter(dim_x=4, dim_z=1)
        self.mape._x = mytargetp +0.
        
        #State transition matrix
        self.mape.F = self.statetransition(dt)   
        self.mape_Fall = []
        self.mape_Fall.append(self.statetransition(dt))
        #Measurement noise
        range_std_map = 0.1
        self.mape.R = np.diag([range_std_map**2])
        #Process noise
        #mape.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.0001)
        #mape.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.0001)
#        self.mape.Q=np.array([[ 0.25,  0.04,  0.  ,  0.  ],
#                         [ 0.04 ,  0.01,  0.  ,  0.  ],
#                         [ 0.  ,  0.  ,  0.25,  0.04],
#                         [ 0.  ,  0.  ,  0.04 ,  0.01]])
        self.mape.Q=np.array([[ 10.,  0.0,  0.  ,  0.  ],
                              [ 0.0 ,  100000.,  0.  ,  0.  ],
                              [ 0.  ,  0.  ,  10.,  0.0],
                              [ 0.  ,  0.  ,  0.0 ,  100000.]])
        #Uncertainty covariance
        self.mape.P *= 1000
        #variables
        self.mape_states = np.matrix(self.mape._x).T
#        self.mape_states = np.matrix([])
        self.all_z =[]
#        self.all_z.append(0)
        self.P_A=[]
#        self.P_A.append([0.,0.])
        self.P_B=[]
#        self.P_B.append([0.,0.])
        
        #MAP2 inisialitzation (marginalizing old states)#####################################################################
        self.mmxs=[]
        self.mm_dt = 0
        self.mm_rtsize = []
        self.aux_index = 0
        self.mape2 = ExtendedKalmanFilter(dim_x=4, dim_z=1)
        self.mape2.x = mytargetp+0.
        #State transition matrix
        self.mape2.F = self.statetransition(dt) 
        self.mape2_Fall = []
        self.mape2_Fall.append(self.statetransition(dt))
        #Measurement noise
        self.mape2.R = np.diag([range_std_map**2])
        #Process noise
        #mape.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.0001)
        #mape.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.0001)
#        self.mape2.Q=np.array([[ 0.25,  0.04,  0.  ,  0.  ],
#                         [ 0.04 ,  0.01,  0.  ,  0.  ],
#                         [ 0.  ,  0.  ,  0.25,  0.04],
#                         [ 0.  ,  0.  ,  0.04 ,  0.01]])
    
#        self.mape2.Q=np.array([[ 10.25,  0.04,  0.  ,  0.  ],
#                         [ 0.04 ,  1000.1,  0.  ,  0.  ],
#                         [ 0.  ,  0.  ,  10.25,  0.04],
#                         [ 0.  ,  0.  ,  0.04 ,  1000.1]])
    
        self.mape2.Q=np.array([[ 10.,  0.,  0.  ,  0.  ],
                               [ 0. ,  100000.,  0.  ,  0.  ],
                               [ 0.  ,  0.  ,  10.,  0.0],
                               [ 0.  ,  0.  ,  0. ,  100000.]])
    
        #Uncertainty covariance
        self.mape2.P *= 1000
        #variables
        self.mape2_states = np.matrix(self.mape2._x).T
        self.mape2_statesaux = np.matrix(self.mape2._x).T
        self.all2_z =[]
#        self.all2_z.append(0)
        self.P_A2=[]
#        self.P_A2.append([0.,0.])
        self.P_B2=[]
#        self.P_B2.append([0.,0.])
        
        ############# WLS initialization ###########################################################################
        self.lsxs=[]
        self.ls_dt = 0
        self.eastingpoints_LS=[]
        self.northingpoints_LS=[]
        self.Plsu=np.array([])
        
        ############# ML initialization ###########################################################################
        self.mlxs=[]
        self.ml_dt = 0
        
        ############# WLS-ML initialization ###########################################################################
        self.lsmlxs=[]
        self.lsml_dt = 0
        

    def statetransition(self,dt):
        F = np.eye(4) + np.array([[0, 1, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 1],
                                  [0, 0, 0, 0]]) * dt
        return F
    
    #target functions    
    def f_target(self,x, dt):
    #    print 'dt=' +str(dt)
        """ state transition function for a
        constant velocity aircraft"""
        F = self.statetransition(dt)
        return np.dot(F, x)
    
    #MAP Functions
    def HJacobian_map(self,x,PA,PB):
        """ compute Jacobian of H matrix at x """
        horiz_dist_a = x[0]-PA[0]
        vert_dist_a  = x[2]-PA[1]
        denom_a = sqrt(horiz_dist_a**2 + vert_dist_a**2)
        horiz_dist_b = x[0]-PB[0]
        vert_dist_b  = x[2]-PB[1]
        denom_b = sqrt(horiz_dist_b**2 + vert_dist_b**2)
        return np.array ([[(horiz_dist_a/denom_a-horiz_dist_b/denom_b)/1500., 0., (vert_dist_a/denom_a - vert_dist_b/denom_b)/1500., 0.]])
    def hx_map(self,x,PA,PB):
        """ compute measurement for slant range that
        would correspond to state x.
        """
        return (((x[0]-PA[0])**2 + (x[2]-PA[1])**2) ** 0.5 - ((x[0]-PB[0])**2 + (x[2]-PB[1])**2) ** 0.5)/1500.

    ######################################################################################
    ###    Particle Filter                                  ##############################
    ######################################################################################
    def updatePF(self,dt,rt,BS):
        '''
        Here we use a Particle Filter (PF) approach to find the target position, which state
        vector is defined as [x,vx,y,vy] and the measurements hare the Time Difference Of Arrival (TDOA)
        '''
        #Because we can have more than one TDOA measurement due to different receptors, we use a loop
        init_time = time.clock()
        numpoints_ext = 0
        for i in range(rt.size-1):
            for ii in range(rt.size-1-i):
                numpoints_ext += 1
                if dt == self.pf_dt:
                        self.pf_dt = dt
                        dt = 0.001
                else:
                    self.pf_dt = dt
                #set observer (landmarks) positions
                position_A = np.array([BS[i].item(0),0.,BS[i].item(1),0.])
                position_B = np.array([BS[ii+i+1].item(0),0.,BS[ii+i+1].item(1),0.])
                #Take new measurement
                z = rt[i]-rt[ii+i+1]
                #Start the PF algorithm uisng observer positions (A-B) and measurement (z)
                max_error = 300.
                # Initialize the particles if needed
                if self.pf.initialized == False:
                    print 'WARNING: Initializing PFs particles'
                    self.pf.init_particles(position_A,position_B, z)
                # Predict step (move all particles)
                self.pf.predict(dt)
                # Update step (weight and resample)
                # Update the weiths according its probability
                self.pf.measurement_prob(z,position_A,position_B)      
                #Resampling        
                self.pf.resampling()
                # Calculate the avarage error. It it's too big the particle filter is initialized                    
                self.pf.evaluation(position_A,position_B,z=z,max_error=max_error)    
                # We compute the average of all particles to fint the target
                self.pf.target_estimation(position_A,position_B)
        #Finally we save only the last target position, which is the sum/average of the different TDOA used
        #Compute PF orientation and save position
        try:
            pf_orientation = np.arctan2(self.pf._x[2]-self.pxs[-1].item(2),self.pf._x[0]-self.pxs[-1].item(0))
        except IndexError:
            pf_orientation = 0
        self.pxs.append(np.append(self.pf._x,pf_orientation))
        pf_time=time.clock()-init_time
        return pf_time

    #############################################################################################
    ####       Batch Maximum A-Posteriori (MAP) Estimator               ########################
    #############################################################################################
    def updateMAP(self,dt,rt,BS):
        '''
        Here we use a Maximum A Posteriory (MAP) approach to find the target position, which state
        vector is defined as [x,vx,y,vy] and the measurements hare the Time Difference Of Arrival (TDOA)
        '''
        #Because we can have more than one TDOA measurement due to different receptors, we use a loop
        init_time = time.clock()
        numpoints_ext = 0
        for i in range(rt.size-1):
            for ii in range(rt.size-1-i):
                numpoints_ext += 1
                #set observer (landmarks) positions
                position_A = np.array([BS[i].item(0),0.,BS[i].item(1),0.])
                position_B = np.array([BS[ii+1+i].item(0),0.,BS[ii+1+i].item(1),0.])
                #append landmarks positions
                self.P_A.append([position_A[0],position_A[2]])
                self.P_B.append([position_B[0],position_B[2]])
                #Take new measurement
                z = rt[i]-rt[ii+1+i]
                #append measurements
                self.all_z.append(z)
        self.m_rtsize.append(numpoints_ext)
        #Propagate current target state estimate
        self.mape.F = self.statetransition(dt)
        self.mape_Fall.append(self.statetransition(dt)) #accumulate all F because each one can have a different dt
        self.mape.predict(dt)
        #Stack all states
        self.mape_states = np.concatenate([self.mape_states,np.matrix(self.mape._x).T],axis=0)
        mape_size = int(self.mape_states.size/4)
                     
        #define NI
        NI = np.concatenate([np.identity(4),np.zeros([4*(mape_size-1),4])]).T
        
        x_l = self.mape_states +0.
        #Start Gauss-Newton Iteration algorithm

        max_num_iterations = 20
        conc_costf = []
        for l in range(max_num_iterations):
            #Compute b(l) of (^X_K)
            # First factor (initial state)
            P_initial=np.matrix(10000000.*np.identity(4))
            initial_error = NI.T*P_initial.I*(x_l[0:4]-self.mape_states[0:4])
            range_error = 0
            process_error = 0
            aux_c = 0
            for k in range(mape_size-1):
                k += 1
                for m in range(self.m_rtsize[k-1]):
                    #Second factor (Range error)
#                    print 'PA=',self.P_A[(k-1)*self.m_rtsize[k-2]+m]
                    H_Me = np.concatenate([np.zeros([4*k,1]), - self.HJacobian_map(np.array(x_l[k*4:(k+1)*4].T)[0],self.P_A[aux_c+m],self.P_B[aux_c+m]).T, np.zeros([4*(mape_size-k-1),1])]).T
                    aux = (self.all_z[aux_c+m])-self.hx_map(np.array(x_l[k*4:(k+1)*4].T)[0],self.P_A[aux_c+m],self.P_B[aux_c+m])
                    range_error += H_Me.T*np.matrix(self.mape.R).I*(aux*1.) 
                aux_c = aux_c + m + 1
                #Third factor (Process error)
                F_Me = np.concatenate([np.zeros([4*(k-1),4]), - self.mape_Fall[k-1].T, np.identity(4), np.zeros([4*(mape_size-k-1),4])]).T
                process_error += F_Me.T*np.matrix(self.mape.Q).I*(x_l[k*4:(k+1)*4]-self.mape_Fall[k-1]*x_l[(k-1)*4:(k)*4]) 
            #the three factors together give b(l)
            b_l = initial_error + range_error + process_error
            
            #Compute A(l) of (^X_K)
            # First factor (initial covariance)
            initial_covariance = NI.T*P_initial.I*NI
            range_covariance = 0
            process_covariance = 0
            aux_c = 0
            for k in range(mape_size-1):
                k += 1
                for m in range(self.m_rtsize[k-1]):
                    #Second factor (Range covariance)
                    H_Me = np.concatenate([np.zeros([4*k,1]), - self.HJacobian_map(np.array(x_l[k*4:(k+1)*4].T)[0],self.P_A[aux_c+m],self.P_B[aux_c+m]).T, np.zeros([4*(mape_size-k-1),1])]).T
                    range_covariance += H_Me.T*np.matrix(self.mape.R).I*H_Me 
                aux_c = aux_c + m + 1
                #Third factor (Process covariance)
                F_Me = np.concatenate([np.zeros([4*(k-1),4]), - self.mape_Fall[k-1].T, np.identity(4), np.zeros([4*(mape_size-k-1),4])]).T
                process_covariance += F_Me.T*np.matrix(self.mape.Q).I*F_Me
            #The three factors together give A(l)
            A_l = initial_covariance + range_covariance + process_covariance
            
            #Compute gX_K(l) (as a linear sistem: A(l) · gX_K(l) = - b(l))
            gx = -A_l.I*b_l
            #Step size (from Portugal PhD)
            sk=0.5
            #Update new state estimation (^X_K) as: [^X_K(l+1)= ^X_K(l) + gXK(l)]
            x_l = x_l + sk*gx
            
            #See if the cost function is lower than the lower limit
            cost_function = np.sqrt(b_l.T*b_l)
            conc_costf.append(cost_function.item(0))

            c_min = 1e-8
            if cost_function <= c_min:
                break

        #update the new state
        k = mape_size -1
        self.mape._x = np.array(x_l[k*4:(k+1)*4].T)[0]
        self.mape_states = x_l + 0.
        #Compute MAP orientation and save position
        try:
            map_orientation = np.arctan2(self.mape._x[2]-self.mape_states[::-1][4:8][::-1].item(2),self.mape._x[0]-self.mape_states[::-1][4:8][::-1].item(0))
        except IndexError:
            map_orientation = 0
#        map_position = np.append(self.mape._x,map_orientation)
#        self.mmxs.append(map_position)
#        Because the MAP redefines all the stacked state vector each time, all the mmxs is updated also each time. Where the mmxs is the predicted target position using MAP
        self.mxs = []
        i = 0
        for i in range(mape_size-1):
            i+=1
            self.mxs.append(np.array([np.array(self.mape_states[0::4].T)[0].item(i),np.array(self.mape_states[1::4].T)[0].item(i),np.array(self.mape_states[2::4].T)[0].item(i),np.array(self.mape_states[3::4].T)[0].item(i),map_orientation]))
        
        map_time=time.clock()-init_time
        return map_time
    
    #############################################################################################
    ####       Batch Maximum A-Posteriori (MAP) Estimator, with Marginalizing old states       ##
    #############################################################################################
    def updateMAPm(self,dt,rt,BS):
        '''
        Here we use a Maximum A Posteriory (MAP) approach to find the target position, which state
        vector is defined as [x,vx,y,vy] and the measurements hare the Time Difference Of Arrival (TDOA)
        '''
        #Because we can have more than one TDOA measurement due to different receptors, we use a loop
        init_time = time.clock()
        #first we need to compute the number of points used
        
        numpoints_ext = 0
        for i in range(rt.size-1):
            for ii in range(rt.size-1-i):
                numpoints_ext += 1
                #set observer (landmarks) positions
                position_A = np.array([BS[i].item(0),0.,BS[i].item(1),0.])
                position_B = np.array([BS[ii+i+1].item(0),0.,BS[ii+i+1].item(1),0.])
                #Take new measurement
                z = rt[i]-rt[ii+i+1]
                #append measurements
                self.all2_z.append(z)
                #append landmarks positions
                self.P_A2.append([position_A[0],position_A[2]])
                self.P_B2.append([position_B[0],position_B[2]])
        self.mm_rtsize.append(numpoints_ext)        
                
        #Propagate current target state estimate
        self.mape2.F = self.statetransition(dt)
        self.mape2_Fall.append(self.statetransition(dt)) #accumulate all F becaurse each one can have a different dt
        self.mape2.predict(dt)
        
        #Stack all states
        self.mape2_states=np.concatenate([self.mape2_states,np.matrix(self.mape2._x).T],axis=0)
        mape2_size = int(self.mape2_states.size/4)
        
        #Marginalizing old states
        swindow = 4
        marginalizing = False

        if mape2_size > swindow:
            self.mape2_states = self.mape2_states[4:]
            self.all2_z = self.all2_z[self.mm_rtsize[self.aux_index]:]
            self.P_A2 = self.P_A2[self.mm_rtsize[self.aux_index]:]
            self.P_B2 = self.P_B2[self.mm_rtsize[self.aux_index]:]
            self.aux_index += 1
            mape2_size = int(self.mape2_states.size/4)
            marginalizing = True

        #define NI
        NI = np.concatenate([np.identity(4),np.zeros([4*(mape2_size-1),4])]).T
        
        x2_l = self.mape2_states +0.
        #Start Gauss-Newton Iteration algorithm
        max_num_iterations = 20
        for l in range(max_num_iterations):
            #Compute b(l) of (^X_K)
            # First factor (initial state)
            if marginalizing == True:
                var = 10000000.
            else:
                var = 10000000.
            P_initial=np.matrix(var*np.identity(4))
            initial_error = NI.T*P_initial.I*(x2_l[0:4]-self.mape2_states[0:4])
            range_error = 0.
            process_error = 0.
            aux_c = 0
            for k in range(mape2_size-1):
                k += 1
                for m in range(self.mm_rtsize[k-1]):
                    #Second factor (Range error)
                    H_Me = np.concatenate([np.zeros([4*k,1]), - self.HJacobian_map(np.array(x2_l[k*4:(k+1)*4].T)[0],self.P_A2[aux_c+m],self.P_B2[aux_c+m]).T, np.zeros([4*(mape2_size-k-1),1])]).T
                    aux = (self.all2_z[aux_c+m])-self.hx_map(np.array(x2_l[k*4:(k+1)*4].T)[0],self.P_A2[aux_c+m],self.P_B2[aux_c+m])
                    range_error += H_Me.T*np.matrix(self.mape2.R).I*(aux*1.)
                aux_c = aux_c + m + 1
                #Third factor (Process error)
                F_Me = np.concatenate([np.zeros([4*(k-1),4]), - self.mape2_Fall[k].T, np.identity(4), np.zeros([4*(mape2_size-k-1),4])]).T
                process_error += F_Me.T*np.matrix(self.mape2.Q).I*(x2_l[k*4:(k+1)*4]-self.mape2_Fall[k]*x2_l[(k-1)*4:(k)*4]) 
            #the three factors together give b(l)
            b_l = initial_error + range_error + process_error*1.

            #Compute A(l) of (^X_K)
            # First factor (initial covariance)
            initial_covariance = NI.T*P_initial.I*NI
            range_covariance = 0.
            process_covariance = 0.
            aux_c = 0
            for k in range(mape2_size-1):
                k += 1
                for m in range(self.mm_rtsize[k-1]):
                    #Second factor (Range covariance)
                    H_Me = np.concatenate([np.zeros([4*k,1]), - self.HJacobian_map(np.array(x2_l[k*4:(k+1)*4].T)[0],self.P_A2[aux_c+m],self.P_B2[aux_c+m]).T, np.zeros([4*(mape2_size-k-1),1])]).T
                    range_covariance += H_Me.T*np.matrix(self.mape2.R).I*H_Me
                aux_c = aux_c + m + 1
                #Third factor (Process covariance)
                F_Me = np.concatenate([np.zeros([4*(k-1),4]), - self.mape2_Fall[k].T, np.identity(4), np.zeros([4*(mape2_size-k-1),4])]).T
                process_covariance += F_Me.T*np.matrix(self.mape2.Q).I*F_Me
            #The three factors together give A(l)
            A_l = initial_covariance + range_covariance + process_covariance*1.
            
            #Compute gX_K(l) (as a linear sistem: A(l) · gX_K(l) = - b(l))
            gx = -A_l.I*b_l
            #Step size (from Portugal PhD)
            sk=0.5
            #Update new state estimation (^X_K) as: [^X_K(l+1)= ^X_K(l) + gXK(l)]
            x2_l = x2_l + sk*gx
            
            #See if the cost function is lower than the lower limit
            cost_function = np.sqrt(b_l.T*b_l)
            c_min = 1e-6
            if cost_function <= c_min:
                break

        #update the new state
        self.mape2._x = np.array(x2_l[k*4:(k+1)*4].T)[0]
        self.mape2_states = x2_l
                
        #Here, we only save the last position obtained with the algorithm, which is the sum/average of all the TDOA measurements
        #Compute MAP orientation and save position
        try:
            map2_orientation = np.arctan2(self.mape2._x[2]-self.mape2_states[::-1][4:8][::-1].item(2),self.mape2._x[0]-self.mape2_states[::-1][4:8][::-1].item(0))
        except IndexError:
            map2_orientation = 0
        #Because the MAP redefines all the stacked state vector each time, all the mmxs is updated also each time. Where the mmxs is the predicted target position using MAP
        if marginalizing == False:
        #if the marginalizing is not performed, we modify all the target predicted psitions (mmxs) as the regurlar procedure
            self.mmxs = []
            i = 0
            for i in range(mape2_size-1):
                i+=1 
                self.mmxs.append(np.array([np.array(self.mape2_states[0::4].T)[0].item(i),np.array(self.mape2_states[1::4].T)[0].item(i),np.array(self.mape2_states[2::4].T)[0].item(i),np.array(self.mape2_states[3::4].T)[0].item(i),map2_orientation]))
        else:
        #if marginalizing is performed, we keep the older value, modify the middle values, and append the last predicted (which increase the vector position mmxs)
            j = self.aux_index +1 
            for i in range(mape2_size-1):
                self.mmxs[j]=np.array([np.array(self.mape2_states[0::4].T)[0].item(i),np.array(self.mape2_states[1::4].T)[0].item(i),np.array(self.mape2_states[2::4].T)[0].item(i),np.array(self.mape2_states[3::4].T)[0].item(i),map2_orientation])
            i += 1
            self.mmxs.append(np.array([np.array(self.mape2_states[0::4].T)[0].item(i),np.array(self.mape2_states[1::4].T)[0].item(i),np.array(self.mape2_states[2::4].T)[0].item(i),np.array(self.mape2_states[3::4].T)[0].item(i),map2_orientation]))
                
        map2_time=time.clock()-init_time
        return map2_time

    #############################################################################################
    ####             Weighted Least Squares Algorithm  (WLS)                                   ##         
    #############################################################################################
    def updateLS(self,dt,rt,bs):
        ''' Using the Least Square (LS) method presented by B. Jin, X. Xu, and T. Zhang
            Title = Robust Time-Difference-of-Arrival (TDOA) Localization Using Weighted Least Squares with Cone Tangent Plane Constraint
            URL = https://www.mdpi.com/1424-8220/18/3/778
        '''
        init_time = time.clock()
        #firstly we eliminate too big easing/northing numbers in order to avoid possible errors
        e_offset = bs[0].item(0)
        n_offset = bs[0].item(1)
        bs.T[0] = bs.T[0] - e_offset
        bs.T[1] = bs.T[1] - n_offset
        
        t_position_5_old = np.matrix([[0],[0],[0]])
        threshold = 0.001
        xs = 1. #initial target estimation position
        ys = 1. #initial target estimation position
        cs = 1500.
        
        for i in range(100):            
            ''' 
            Target position estimation
            '''
            Ps = np.matrix([[xs],
                            [ys]])
            
            '''
            How to compute the difference distances (d) using the times reception input array 
            d = np.matrix([[cs*(t2t-t1t)],
                           [cs*(t3t-t1t)],
                           [cs*(t4t-t1t)]])
            '''
            d = np.matrix(rt-rt[0]).T[1:]*cs
            
            '''
            How to compute the Gb matrix automatically and independently the number of base stations (bs) and receptions used
            Gb = np.matrix([[A2[0]-A1[0], A2[1]-A1[1], d.item(0)],
                            [A3[0]-A1[0], A3[1]-A1[1], d.item(1)],
                            [A4[0]-A1[0], A4[1]-A1[1], d.item(2)],
                            [Ps[0]-A1[0], Ps[1]-A1[1], -np.sqrt(((Ps.T-np.matrix(A1[:2]))*(Ps.T-np.matrix(A1[:2])).T).item(0))],
                            [Ps[1],      -Ps[0]      , 0. ]])
            '''
            Gb_aux1 = (bs[:,:-1]-bs[:,:-1][0])[1:]
            Gb_aux2 = np.append(Gb_aux1,d,axis=1) 
            Gb_aux3 = np.matrix([[(Ps[0]-bs[0][0]).item(0), (Ps[1]-bs[0][1]).item(0), -np.sqrt(((Ps.T-np.matrix(bs[0][:2]))*(Ps.T-np.matrix(bs[0][:2])).T).item(0)).item(0)],
                                 [Ps.item(1),      -Ps.item(0)      , 0. ]])
            Gb      = np.append(Gb_aux2,Gb_aux3,axis=0) 
            
            '''
            Weight matrix (w) of the Weighted Least Square algorithm (WLS)
            '''
            r = np.sqrt(np.diag((bs[:,:2].T - Ps).T * (bs[:,:2].T - Ps)))
            Ba = np.diag(np.sqrt(np.diag((bs[:,:2].T - Ps).T * (bs[:,:2].T - Ps))))
            std_noise = 0.0001
            Q = np.identity(4)*std_noise**2
            w_ma = np.diag(cs**2*Ba*Q*Ba)[1:]
            w_va = r[0]*0.45
            w_ca = r[0]*0.45**2/2.
            w = np.matrix(np.diag(np.append(w_ma,[w_ca**2,w_va**2])))
    
            '''
            How to compute the hb matirx automatically and independently the number of base stations (bs) and receptions used
            hb = 0.5*np.matrix([[((A2[0]**2+A2[1]**2)-(A1[0]**2+A1[1]**2)-d[0]**2).item(0)],
                                [((A3[0]**2+A3[1]**2)-(A1[0]**2+A1[1]**2)-d[1]**2).item(0)],
                                [((A4[0]**2+A4[1]**2)-(A1[0]**2+A1[1]**2)-d[2]**2).item(0)],
                                [2*(Ps-np.matrix(A1[:2]).T).T*np.matrix(A1[:2]).T],
                                [0]])
            '''
            hb_aux1 = np.matrix(np.diag(np.matrix(bs[:,:-1])*np.matrix(bs[:,:-1]).T)).T
            hb_aux2 = (hb_aux1-hb_aux1[0])[1:]-np.matrix(np.diag(d*d.T)).T
            hb_aux3 = np.matrix([[(2*(Ps-np.matrix(bs[0][:2]).T).T*np.matrix(bs[0][:2]).T).item(0)],
                                 [0]])
            hb      = 0.5 * np.append(hb_aux2,hb_aux3,axis=0)
            
            '''
            Compute the target position
            '''
            t_position_5 = (Gb.T*w.I*Gb).I*Gb.T*w.I*hb 
            
            '''
            If the new target position estimation is lower than the threshold, the we break the loop and set the target position
            as the latest estimation
            '''
            difference = (t_position_5[:2]-t_position_5_old[:2]).T*(t_position_5[:2]-t_position_5_old[:2])
            if difference < threshold:
                break
            t_position_5_old = t_position_5 + 0.
            xs = t_position_5.item(0) #updated target estimation position
            ys = t_position_5.item(1) #updated target estimation position
        #Compute LS orientation and save position
        try:
            cc = t_position_5.item(0)+e_offset-self.lsxs[-1][0]
            co = t_position_5.item(1)+n_offset-self.lsxs[-1][2]
            if cc == 0. and co == 0.:
                ls_orientation = self.lsxs[-1].item(4)
            else:
                ls_orientation = np.arctan2(co,cc)
        except IndexError:
            ls_orientation = 0
        try:
            ls_velocity = np.array([(t_position_5.item(0)+e_offset-self.lsxs[-1][0])/dt,(t_position_5.item(1)+n_offset-self.lsxs[-1][2])/dt])
        except IndexError:
            ls_velocity = np.array([0,0])
        ls_position = np.array([t_position_5.item(0)+e_offset,ls_velocity[0],t_position_5.item(1)+n_offset,ls_velocity[1],ls_orientation])
        self.lsxs.append(ls_position)
        ls_time=time.clock()-init_time
        return ls_time
    
    #############################################################################################
    ####             Maximum Likelihood Estimation  (ML)                                       ##         
    #############################################################################################
    def updateML(self,dt,rt,BS):
        '''
        Using Maximum Likelihood Estimation method 2D, and all possible TDOA
        '''
        init_time = time.clock()
        cs = 1500.
        # 1: Start at initial estimate###################################
        k = 0
        error = 1.
        xx = np.matrix([[0.],[0.]])
        numpoints = rt.size-1
        std = error+0.
        d = np.matrix([])
        for i in range(numpoints):
            try :
                d = np.append(d,np.matrix(rt[i]-rt).T[i+1:]*cs,axis=0) #different distances
            except:
                #its first time so
                d = np.matrix(rt[i]-rt).T[i+1:]*cs
        while(1):
            #2: Determine a search direction ################################
            #gradien method
            P_A = np.matrix([])
            numpoints_ext = 0
            for i in range(numpoints):
                for j in range(numpoints-i):
                    numpoints_ext +=1
                    if i == 0 and j == 0:
                        #its first time so:
                        P_A = np.matrix(BS[i][:2]).T
                    else:
                        P_A = np.append(P_A,np.matrix(BS[i][:2]).T,axis=1)
            P_B = np.matrix([])
            for i in range(numpoints):
                for j in range(numpoints-i):
                    if i == 0 and j == 0:
                        #its first time so:
                        P_B = np.matrix(BS[j+1+i][:2]).T
                    else:
                        P_B = np.append(P_B,np.matrix(BS[j+1+i][:2]).T,axis=1) 
            Cx_A = xx*np.ones(numpoints_ext) - P_A
            Cx_B = xx*np.ones(numpoints_ext) - P_B
            variance_ML = std**2
            Rx = np.matrix(variance_ML*np.identity(numpoints_ext))
            rx1_A = xx-P_A
            rx2_A = np.matrix(np.sqrt(np.diag(rx1_A.T*rx1_A))).T
            grx_A = np.matrix(np.array(rx2_A.T)*np.identity(numpoints_ext)) 
            rx1_B = xx-P_B
            rx2_B = np.matrix(np.sqrt(np.diag(rx1_B.T*rx1_B))).T
            grx_B = np.matrix(np.array(rx2_B.T)*np.identity(numpoints_ext)) 
            r_rx_AB = d - (rx2_A-rx2_B)
            try:
                gradient = -(Cx_A*grx_A.I-Cx_B*grx_B.I)*Rx.I*r_rx_AB
            except:
                grx_A += np.identity(grx_A[0].size)*1e-12
                try:
                    gradient = -(Cx_A*grx_A.I-Cx_B*grx_B.I)*Rx.I*r_rx_AB
                except:
                    grx_B += np.identity(grx_B[0].size)*1e-12
                    gradient = -(Cx_A*grx_A.I-Cx_B*grx_B.I)*Rx.I*r_rx_AB                    
            hx = -gradient
            
            #newtwon method
            alpha = Rx.I*r_rx_AB
            g_alpha = np.matrix(np.array(alpha.T)*np.identity(numpoints_ext)) 
            aux_A = (alpha.T*grx_A.I*np.matrix(np.ones(numpoints_ext)).T).item(0)*np.identity(2)-Cx_A*(grx_A**3).I*g_alpha*Cx_A.T
            aux_B = (alpha.T*grx_B.I*np.matrix(np.ones(numpoints_ext)).T).item(0)*np.identity(2)-Cx_B*(grx_B**3).I*g_alpha*Cx_B.T
            gradient2 = (Cx_A*grx_A.I-Cx_B*grx_B.I)*Rx.I*(grx_A.I*Cx_A.T-grx_B.I*Cx_B.T) + (aux_A-aux_B) 
            try:
                hx = -gradient2.I*gradient
            except:
                gradient2 += np.identity(gradient2[0].size)*1e-12
                hx = -gradient2.I*gradient
                
            #3: Determine step size sk, using Armijo rule ##################
            s = 1.
            B = 1./2. #between 1/2 - 1/10
            a = 1e-1 #between 1e-1 - 1e-5
            m = 0.
            e = 1e-8 #0.000005
            while(1):
                rx11_A = (xx+s*B**m*hx)-P_A
                rx22_A = np.matrix(np.sqrt(np.diag(rx11_A.T*rx11_A))).T
                rx11_B = (xx+s*B**m*hx)-P_B
                rx22_B = np.matrix(np.sqrt(np.diag(rx11_B.T*rx11_B))).T
                r_rx2_AB = np.matrix(d) - (rx22_A-rx22_B)
                if 0.5*r_rx2_AB.T*Rx.I*r_rx2_AB <= (0.5*r_rx_AB.T*Rx.I*r_rx_AB + a*s*B**m*hx.T*gradient):
                    break
                else:
                    m = m+1
            sk = s*B**m
            # 4: Update estimate
            xx = xx + sk*hx
            k=k+1
            # 5: next or stop
            if k > 40 or np.sqrt(gradient.T*gradient).item(0)<e:
                break
        #Compute ML orientation and save position
        try:
            cc = xx.item(0)-self.mlxs[-1][0]
            co = xx.item(1)-self.mlxs[-1][2]
            if cc == 0. and co == 0.:
                ls_orientation = self.mlxs[-1].item(4)
            else:
                ls_orientation = np.arctan2(co,cc)
        except IndexError:
            ls_orientation = 0
        try:
            ls_velocity = np.array([(xx.item(0)-self.mlxs[-1][0])/dt,(xx.item(1)-self.mlxs[-1][2])/dt])
        except IndexError:
            ls_velocity = np.array([0,0])
        ls_position = np.array([xx.item(0),ls_velocity[0],xx.item(1),ls_velocity[1],ls_orientation])
        self.mlxs.append(ls_position)
        ml_time=time.clock()-init_time
        return ml_time
    
    #############################################################################################
    ####  Weighted Least Squares Algorithm  (WLS) and Maximum Likelihood (ML) Estimation       ##         
    #############################################################################################
    def updateLSMLE(self,dt,rt,bs):
        ''' Using the Least Square (LS) method presented by B. Jin, X. Xu, and T. Zhang
            Title = Robust Time-Difference-of-Arrival (TDOA) Localization Using Weighted Least Squares with Cone Tangent Plane Constraint
            URL = https://www.mdpi.com/1424-8220/18/3/778
        '''
        init_time = time.clock()
        #firstly we eliminate too big easing/northing numbers in order to avoid possible errors
        e_offset = bs[0].item(0)
        n_offset = bs[0].item(1)
        bs.T[0] = bs.T[0] - e_offset
        bs.T[1] = bs.T[1] - n_offset
        
        t_position_5_old = np.matrix([[0],[0],[0]])
        threshold = 0.001
        xs = 1. #initial target estimation position
        ys = 1. #initial target estimation position
        cs = 1500.
        
        for i in range(100):
        #    xs = -399.092162
        #    ys = -159.567475
            
            ''' 
            Target position estimation
            '''
            Ps = np.matrix([[xs],
                            [ys]])
            
            '''
            How to compute the difference distances (d) using the times reception input array 
            d = np.matrix([[cs*(t2t-t1t)],
                           [cs*(t3t-t1t)],
                           [cs*(t4t-t1t)]])
            '''
            d = np.matrix(rt-rt[0]).T[1:]*cs
            
            '''
            How to compute the Gb matrix automatically and independently the number of base stations (bs) and receptions used
            Gb = np.matrix([[A2[0]-A1[0], A2[1]-A1[1], d.item(0)],
                            [A3[0]-A1[0], A3[1]-A1[1], d.item(1)],
                            [A4[0]-A1[0], A4[1]-A1[1], d.item(2)],
                            [Ps[0]-A1[0], Ps[1]-A1[1], -np.sqrt(((Ps.T-np.matrix(A1[:2]))*(Ps.T-np.matrix(A1[:2])).T).item(0))],
                            [Ps[1],      -Ps[0]      , 0. ]])
            '''
            Gb_aux1 = (bs[:,:-1]-bs[:,:-1][0])[1:]
            Gb_aux2 = np.append(Gb_aux1,d,axis=1) 
            Gb_aux3 = np.matrix([[(Ps[0]-bs[0][0]).item(0), (Ps[1]-bs[0][1]).item(0), -np.sqrt(((Ps.T-np.matrix(bs[0][:2]))*(Ps.T-np.matrix(bs[0][:2])).T).item(0)).item(0)],
                                 [Ps.item(1),      -Ps.item(0)      , 0. ]])
            Gb      = np.append(Gb_aux2,Gb_aux3,axis=0) 
            
            '''
            Weight matrix (w) of the Weighted Least Square algorithm (WLS)
            '''
            r = np.sqrt(np.diag((bs[:,:2].T - Ps).T * (bs[:,:2].T - Ps)))
            Ba = np.diag(np.sqrt(np.diag((bs[:,:2].T - Ps).T * (bs[:,:2].T - Ps))))
            std_noise = 0.0001
            Q = np.identity(4)*std_noise**2
            w_ma = np.diag(cs**2*Ba*Q*Ba)[1:]
            w_va = r[0]*10.45
            w_ca = r[0]*10.45**2/2.
            w = np.matrix(np.diag(np.append(w_ma,[w_ca**2,w_va**2])))
    
            '''
            How to compute the hb matirx automatically and independently the number of base stations (bs) and receptions used
            hb = 0.5*np.matrix([[((A2[0]**2+A2[1]**2)-(A1[0]**2+A1[1]**2)-d[0]**2).item(0)],
                                [((A3[0]**2+A3[1]**2)-(A1[0]**2+A1[1]**2)-d[1]**2).item(0)],
                                [((A4[0]**2+A4[1]**2)-(A1[0]**2+A1[1]**2)-d[2]**2).item(0)],
                                [2*(Ps-np.matrix(A1[:2]).T).T*np.matrix(A1[:2]).T],
                                [0]])
            '''
            hb_aux1 = np.matrix(np.diag(np.matrix(bs[:,:-1])*np.matrix(bs[:,:-1]).T)).T
            hb_aux2 = (hb_aux1-hb_aux1[0])[1:]-np.matrix(np.diag(d*d.T)).T
            hb_aux3 = np.matrix([[(2*(Ps-np.matrix(bs[0][:2]).T).T*np.matrix(bs[0][:2]).T).item(0)],
                                 [0]])
            hb      = 0.5 * np.append(hb_aux2,hb_aux3,axis=0)
            
            '''
            Compute the target position
            '''
            t_position_5 = (Gb.T*w.I*Gb).I*Gb.T*w.I*hb 
            
            '''
            If the new target position estimation is lower than the threshold, the we break the loop and set the target position
            as the latest estimation
            '''
            difference = (t_position_5[:2]-t_position_5_old[:2]).T*(t_position_5[:2]-t_position_5_old[:2])
            if difference < threshold:
                break
            t_position_5_old = t_position_5 + 0.
            xs = t_position_5.item(0) #updated target estimation position
            ys = t_position_5.item(1) #updated target estimation position
            
            
        '''
        Using Maximum Likelihood Estimation method 2D, and all possible TDOA where the initial guest is the positio found with LS
        '''
        # 1: Start at initial estimate###################################
        k = 0
        error = 1.
        xx = np.matrix([[t_position_5.item(0)],[t_position_5.item(1)]])
        numpoints = rt.size-1
        std = error+0.
        d = np.matrix([])
        for i in range(numpoints):
            try :
                d = np.append(d,np.matrix(rt[i]-rt).T[i+1:]*cs,axis=0) #different distances
            except:
                #its first time so
                d = np.matrix(rt[i]-rt).T[i+1:]*cs
        while(1):
            #2: Determine a search direction ################################
            #gradien method
            P_A = np.matrix([])
            numpoints_ext = 0
            for i in range(numpoints):
                for j in range(numpoints-i):
                    numpoints_ext +=1
                    if i == 0 and j == 0:
                        #its first time so:
                        P_A = np.matrix(bs[i][:2]).T
                    else:
                        P_A = np.append(P_A,np.matrix(bs[i][:2]).T,axis=1)
            P_B = np.matrix([])
            for i in range(numpoints):
                for j in range(numpoints-i):
                    if i == 0 and j == 0:
                        #its first time so:
                        P_B = np.matrix(bs[j+1+i][:2]).T
                    else:
                        P_B = np.append(P_B,np.matrix(bs[j+1+i][:2]).T,axis=1) 
            Cx_A = xx*np.ones(numpoints_ext) - P_A
            Cx_B = xx*np.ones(numpoints_ext) - P_B
            variance_ML = std**2
            Rx = np.matrix(variance_ML*np.identity(numpoints_ext))
            rx1_A = xx-P_A
            rx2_A = np.matrix(np.sqrt(np.diag(rx1_A.T*rx1_A))).T
            grx_A = np.matrix(np.array(rx2_A.T)*np.identity(numpoints_ext)) 
            rx1_B = xx-P_B
            rx2_B = np.matrix(np.sqrt(np.diag(rx1_B.T*rx1_B))).T
            grx_B = np.matrix(np.array(rx2_B.T)*np.identity(numpoints_ext)) 
            r_rx_AB = d - (rx2_A-rx2_B)
            try:
                gradient = -(Cx_A*grx_A.I-Cx_B*grx_B.I)*Rx.I*r_rx_AB
            except:
                grx_A += np.identity(grx_A[0].size)*1e-12
                try:
                    gradient = -(Cx_A*grx_A.I-Cx_B*grx_B.I)*Rx.I*r_rx_AB
                except:
                    grx_B += np.identity(grx_B[0].size)*1e-12
                    gradient = -(Cx_A*grx_A.I-Cx_B*grx_B.I)*Rx.I*r_rx_AB
                    
            hx = -gradient
            
            #newtwon method
            alpha = Rx.I*r_rx_AB
            g_alpha = np.matrix(np.array(alpha.T)*np.identity(numpoints_ext)) 
            aux_A = (alpha.T*grx_A.I*np.matrix(np.ones(numpoints_ext)).T).item(0)*np.identity(2)-Cx_A*(grx_A**3).I*g_alpha*Cx_A.T
            aux_B = (alpha.T*grx_B.I*np.matrix(np.ones(numpoints_ext)).T).item(0)*np.identity(2)-Cx_B*(grx_B**3).I*g_alpha*Cx_B.T
            gradient2 = (Cx_A*grx_A.I-Cx_B*grx_B.I)*Rx.I*(grx_A.I*Cx_A.T-grx_B.I*Cx_B.T) + (aux_A-aux_B) 
            try:
                hx = -gradient2.I*gradient
            except:
                gradient2 += np.identity(gradient2[0].size)*1e-12
                hx = -gradient2.I*gradient
                
            #3: Determine step size sk, using Armijo rule ##################
            s = 1.
            B = 1./2. #between 1/2 - 1/10
            a = 1e-1 #between 1e-1 - 1e-5
            m = 0.
            e = 1e-8 #0.000005
            while(1):
                rx11_A = (xx+s*B**m*hx)-P_A
                rx22_A = np.matrix(np.sqrt(np.diag(rx11_A.T*rx11_A))).T
                rx11_B = (xx+s*B**m*hx)-P_B
                rx22_B = np.matrix(np.sqrt(np.diag(rx11_B.T*rx11_B))).T
                r_rx2_AB = np.matrix(d) - (rx22_A-rx22_B)
                if 0.5*r_rx2_AB.T*Rx.I*r_rx2_AB <= (0.5*r_rx_AB.T*Rx.I*r_rx_AB + a*s*B**m*hx.T*gradient):
                    break
                else:
                    m = m+1
            sk = s*B**m
            # 4: Update estimate
            xx = xx + sk*hx
            k=k+1
            # 5: next or stop
            if k > 40 or np.sqrt(gradient.T*gradient).item(0)<e:
                break
        #Compute ML orientation and save position
        try:
            cc = xx.item(0)-self.lsmlxs[-1][0]
            co = xx.item(1)-self.lsmlxs[-1][2]
            if cc == 0. and co == 0.:
                ls_orientation = self.lsmlxs[-1].item(4)
            else:
                ls_orientation = np.arctan2(co,cc)
        except IndexError:
            ls_orientation = 0
        try:
            ls_velocity = np.array([(xx.item(0)-self.lsmlxs[-1][0])/dt,(xx.item(1)-self.lsmlxs[-1][2])/dt])
        except IndexError:
            ls_velocity = np.array([0,0])
        ls_position = np.array([xx.item(0)+e_offset,ls_velocity[0],xx.item(1)+n_offset,ls_velocity[1],ls_orientation])
        self.lsmlxs.append(ls_position)
        lsml_time=time.clock()-init_time
        return lsml_time
    
        
        
        
