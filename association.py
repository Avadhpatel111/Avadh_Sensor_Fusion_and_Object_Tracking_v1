# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import math
from scipy.stats import chi2


import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        association_matrix = []
        num_tracks = len(track_list)
        num_meas = len(meas_list)
        
        for track in track_list:
            list_actual_association = [] #create an empty list for actual association matrix
            for meas in meas_list:
            #using MHD method to find the Mahalanobis distance
                Mahalanobis_Distance = self.MHD(track, meas, KF)

                if self.gating(Mahalanobis_Distance, meas.sensor):
                # add the actual value of Mahalanobis_Distance to the list
                    list_actual_association.append(Mahalanobis_Distance)
                else:
                # add a very high infinity value of numpy
                    list_actual_association.append(np.inf)
            association_matrix.append(list_actual_association)
     
        #update the unassigned tracks as a list of total tracks
        self.unassigned_tracks = list(range(num_tracks))
        #update the unassigned meas as a list of total meas
        self.unassigned_meas = list(range(num_meas))
        
        self.association_matrix = np.matrix(association_matrix)
        
        return
        
		
		# the following only works for at most one track and one measurement
      #  association_matrix = []
      #  self.unassigned_tracks = [] # reset lists
      #  self.unassigned_meas = []
#         
#         if len(meas_list) > 0:
#             self.unassigned_meas = [0]
#         if len(track_list) > 0:
#             self.unassigned_tracks = [0]
#         if len(meas_list) > 0 and len(track_list) > 0: 
#             self.association_matrix = np.matrix([[0]])


        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ###########
		
        temp_association_mat = self.association_matrix
		
		#if the all are infinity, none are found
        if np.min(temp_association_mat) == np.inf:
            return np.nan, np.nan

        # to retrieve the indices of minimum entry
        ij_min = np.unravel_index(np.argmin(temp_association_mat, axis=None), temp_association_mat.shape) 
        
        track_indices, meas_indices = ij_min[0], ij_min[1]

        # corresponding index element can be deleted from the row and column for next update
        temp_association_mat = np.delete(temp_association_mat, track_indices, 0) 
        temp_association_mat = np.delete(temp_association_mat, meas_indices, 1)
		
        self.association_matrix = temp_association_mat

        # track is updated with that measurement values
		
        update_track = self.unassigned_tracks[track_indices] 
        update_meas = self.unassigned_meas[meas_indices]
		
		
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)

            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
		#Thresold for gating is set
        gating_thresold = 0.9942
		#setting the limit
        df = sensor.dim_meas #corrected as per reviewer comment
        #print("printing sensor dim here")
        #print(sensor.dim_meas)
        limit = chi2.ppf(gating_thresold, df)
		#comparing Mahalanobis Distance with the limit to set the boolean flag
        
        comp = MHD < limit        #returning the set flag
        return comp   
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
    
        gamma_temp = meas.z - (meas.sensor.get_hx(track.x)) 
        temp_S = meas.R
        
        Mahalanobis_Distance = math.sqrt(gamma_temp.T * temp_S.I * gamma_temp)
        
		#print("printing Mahalanobis_Distance here")
		#print(Mahalanobis_Distance)
		
        return Mahalanobis_Distance
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)