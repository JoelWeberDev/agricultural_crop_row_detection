"""
Author: Joel Weber
Title: Line Matching
Overview: This is the unifying portion that gets all the averaged lines and calculates the most probable line location for the crop rows to be
Vision: This will accurately predict the rows based on the multiple lines from the incremental mask
Notes: This can also work on a frame that only has detected lines on an individual frame
Potential Problems: The provided values and calculations must be quite accurate to detect the lines on the centers of the rows. Perhaps we can calculate the ideal while still keeping the loacation of the 
green detected line to represent the actual detection. 

"""
import numpy as np
import cv2
import math
import sys,os



sys.path.append(os.path.abspath(os.path.join('.')))

from Adaptive_params.Adaptive_parameters import param_manager as ap
# from Modules.Camera_Calib.

"""
INPUTS:
 - an iteable of lines
User Values:
 - row number
 - row spacing
 - row width

Outline: 
 1. Calc the ideal spacing of the pixles

"""
class line_match(object):
    def __init__(self, lines):
        self.ap = ap() 
        self.lines = lines
        self.row_num = self.ap.access("row_count")
        self.spacing = self.ap.access("spacing")
        self.width = self.ap.access("row_width")

        self.calc_ideal_spacing()
        self.calc_ideal_row_width()
        self.calc_ideal_row_spacing()

    # all the user values will be in a world measure and will default to centimeters
    def calc_ideal_spacing(self):
        # Consider this done for not, but we will need to again get into the camera calcultions to determine what the spacing is going to be like.
        # In the end we are hoping to make the stero vision do the calculations for it will be the fastest and the most accurate calculation 

        pass

    def calc_hub(self, lines):
        # Iterate through the lines 
        pass

    """
    This develops the function to score the distance from the ideal that the line is located 
    It uses the following values to develop the score
    - idealized location
    - the row width
    - an adaptive constant

    take the x_value of the line that is being evaluated and determine if it exists on a discrete multiple of widths from the base line

    """
    def _distance_hur(self, pos,rows_in_img = 5):
        # the coeff is the constant that usually is around 0.5 
        self.width*(self.ap.access("inlier_coefficient")+(1/rows_in_img))




        


