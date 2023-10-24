# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import logging
# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = params.dim_state # process model dimension
        self.dt = params.dt # time increment
        self.q = params.q # process noise variable for Kalman filter Q


    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        #dim = self.dim_state
        F = np.identity(self.dim_state)
        #F = F.reshape(dim,dim)
        F[0, 3] = self.dt 
        F[1, 4] = self.dt 
        F[2, 5] = self.dt
		
        return np.matrix(F)
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        
        q = params.q
        dt = params.dt
        q1 = ((dt**3)/3) * q 
        q2 = ((dt**2)/2) * q 
        q3 = dt * q 
        return np.matrix([[q1, 0, 0, q2, 0, 0],
                          [0, q1, 0, 0, q2, 0],
                          [0, 0, q1, 0, 0, q2],
                          [q2, 0, 0, q3, 0, 0],
                          [0, q2, 0, 0, q3, 0],
                          [0, 0, q2, 0, 0, q3]])
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        F_Mat = self.F()
		# the state is given by
        x = F_Mat * track.x
		#for predicting the co-variants
        P = (F_Mat * track.P * F_Mat.transpose()) + self.Q()
        track.set_x(x)
        track.set_P(P)
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        temp_P = track.P
        temp_x = track.x
        
        # matrix to measure
        H = meas.sensor.get_H(temp_x)
        #residual calculation
        gamma = self.gamma(track, meas)
        #covar of res
        S = self.S(track, meas, H)
        #print("Kamlman Gain")
        
        K = temp_P * H.transpose()* S.I
        
        #print(K)
        #print(temp_x)
        temp_x = temp_x + K * gamma
        #print(temp_x)
        #create identity matrix
        I = np.identity(self.dim_state)
        
        #update for the covariant
        temp_P = (I - K * H) * temp_P
        
        track.set_x(temp_x)
        track.set_P(temp_P)
        track.update_attributes(meas)
        ############
        # END student code
        ############ 
        
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############        
        
        return meas.z - meas.sensor.get_hx(track.x)        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        return ((H * track.P * H.transpose()) + meas.R)        
        ############
        # END student code
        ############ 