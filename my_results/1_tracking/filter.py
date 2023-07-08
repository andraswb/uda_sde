# flake8: noqa
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

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
sys.path.append(os.path.normpath(SCRIPT_DIR))
from misc.params import dt, q, dim_state
from trackmanagement import Track
from measurements import Measurement

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############        
        
        return np.matrix(
            [
                [1, 0, 0, dt,  0,  0],
                [0, 1, 0,  0, dt,  0],
                [0, 0, 1,  0,  0, dt],
                [0, 0, 0,  1,  0,  0],
                [0, 0, 0,  0,  1,  0],
                [0, 0, 0,  0,  0,  1],
            ]
        )
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############

        # do it like in exercise
        q1 = ((dt**3)/3) * q
        q2 = ((dt**2)/2) * q
        q3 = dt * q
        return np.matrix(
            [
                [q1, 0, 0, q2, 0, 0],
                [0, q1, 0, 0, q2, 0],
                [0, 0, q1, 0, 0, q2],
                [q2, 0, 0, q3, 0, 0],
                [0, q2, 0, 0, q3, 0],
                [0, 0, q2, 0, 0, q3],
            ]
        )
        
        ############
        # END student code
        ############ 

    def predict(self, track: Track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        
        F = self.F()
        x = F * track.x # state prediction
        P = F * track.P * F.transpose() + self.Q() # covariance prediction
        
        track.set_x(x)
        track.set_P(P)
        
        ############
        # END student code
        ############ 

    def update(self, track: Track, meas: Measurement):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        
        H = meas.sensor.get_H(track.x) # measurement matrix
        
        gamma = self.gamma(track, meas) # residual
        
        # covariance of residual
        S = self.S(track, meas, H)
        # Kalman gain
        K = track.P * H.transpose()* np.linalg.inv(S) 
        # state update
        x = track.x + K * gamma
        track.set_x(x)
        # covariance update
        I = np.identity(dim_state)
        P = (I - K * H) * track.P
        track.set_P(P)
        
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track: Track, meas: Measurement):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############

        if meas.sensor.name == 'lidar':
            H = meas.sensor.get_H(track.x)
            gamma = meas.z - H * track.x
        elif meas.sensor.name == 'camera':
            hx = meas.sensor.get_hx(track.x)
            gamma = meas.z - hx
        else:
            raise RuntimeError(f"Unknown sensor name: {meas.sensor.name}")

        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track: Track, meas: Measurement, H: np.matrix):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        S = H * track.P * H.transpose() + meas.R # covariance of residual
        
        return S
        
        ############
        # END student code
        ############ 