# Calibration_solver

import numpy as np
import math 
import cv2

# Local libraries:
import Camera_Calibration as cc


class camera_calib(object):
    def __init__(self, intrisic=None, extrinsic=None, distortion=None):
        if intrisic is None:
            intrisic = np.zeros((3,3))
        if extrinsic is None:
            extrinsic = np.zeros((3,4))
        self.intrisic = intrisic
        self.extrinsic = extrinsic
        self.distortion = distortion

    def get_intrisic(self):
        
        return self.intrisic 