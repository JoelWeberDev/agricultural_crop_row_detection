"""
Author: Joel Weber
Date: 19/07/2023
Title: Stong_horz_kernel
Type: experimental

Description: Here we perform testing that applies a different kernels to the image to determine if there is a way to get the center of a row more accurately with less processing of the lines themselves. 
 This could also accerlate the line detection through having a more granular image at the time we run the hough line. 

Test ideas:
 1. apply a strong horizontal kernal about the width of a row to each masked image to hopefully get the centers of the rows 
 2. Resume the RANSAC detection rather than the hough line detection for a more systematic approach to the line detection *Note this may have possible problems dealing with vainshing point errors and also spacing issues
 3. 
"""

import cv2
import numpy as np
import sys,os
from icecream import ic
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join('.')))
from Adaptive_params import Adaptive_parameters as ap
from Modules.prep_input import interpetInput as prep 
from Modules import display_frames as disp
from Modules.Image_Processor import apply_funcs as pre_process 


class kerneling(object):
    def __init__(self, img):
        self.img = img
        self.vert_coeff = 0.05
        self.horz_coeff = 0.05
        # self.vert_kern = np.ones((1,int(self.vert_coeff*self.img.shape[1])),np.uint8)
        # self.horz_kern = np.ones((int(self.horz_coeff*self.img.shape[0]),1),np.uint8)
        self.vert_kern = np.ones((1,50),np.uint8)
        self.horz_kern = np.ones((50,1),np.uint8)

    def apply_kernel(self):
        # Apply the kernels to the image

        vert_img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, self.vert_kern)
        horz_img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, self.horz_kern)

        # disp().show(vert_img)
        # disp().show(horz_img)

        return vert_img,horz_img

def show(imgs):
    if type(imgs) == list:
        im_dict = {}
        for i, img in enumerate(imgs):
            im_dict[f"img{i}"] = img
    else: 
        assert type(imgs) == dict, "The input must be either a list or a dictionary"
        im_dict = imgs
    disp.dispMat(im_dict)

    

class test(object):
    def __init__(self):
        self.test_paths = [
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\corn\\imgs",
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\Drone_images\\Winter Wheat",
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\sm_soybeans\\imgs",
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\sm_soybeans\\Quite_weedy"
        ]

    def run(self, func):
        for path in self.test_paths:
            disp.b = False
            self.data = prep('sample', path)["imgs"]
            for img in self.data:
                if disp.b == True: break
                func(img)

    def no_proc(self, img):
        show([img])
    
    def kernel_img(self, img):
        kernel = kerneling(img)
        vert_img,horz_img = kernel.apply_kernel()

        show({"org":img,"vert":vert_img,"horz":horz_img,"comb":vert_img+horz_img})

    
        

if __name__ == "__main__":
    tests = test()
    tests.run(tests.kernel_img)