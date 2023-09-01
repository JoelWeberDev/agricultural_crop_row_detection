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

How could I make a better grouping method of similar pixels?  How do I ensure that the line is in the center of the pixel density? Should I do some sort of cumulative sum
that I the center of the the pixel density is? 
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
from Algorithims_tst.MAKE_SOME_NOISE import noise 

"""
Determine the size of the horizontal kernel based on the size of the image along with the row width and row spacign 
"""
class kerneling(object):
    def __init__(self):
        # self.vert_kern = np.ones((1,int(self.vert_coeff*self.img.shape[1])),np.uint8)
        # self.horz_kern = np.ones((int(self.horz_coeff*self.img.shape[0]),1),np.uint8)
        self.params = ap.param_manager()

    def apply_kernel(self,img,**kwargs):
        # Apply the kernels to the image

        self.img = img
        ret_imgs = {"org":self.img}
        app_mask = kwargs.get("apply_mask",False)
        app_noise = kwargs.get("apply_noise",False)
        disp_cmd = kwargs.get("disp_cmd",None)
        row_width = self.params.access("avg_bot_width_pxl")
        spacing = self.params.access("avg_spacing_pxls")

        self.calc_kernel_size()
        
        assert type(app_mask) == bool, "The apply_mask parameter must be a boolean"
        assert type(app_noise) == bool, "The apply_noise parameter must be a boolean"

        if app_mask:
            mask = self.apply_mask()
            ret_imgs["mask"] = mask
        if app_noise:
            noised = self.add_noise()
            ret_imgs["noised"] = noised


        # vert_img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, self.vert_kern); ret_imgs["vert"] = vert_img
        horz_img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, self.horz_kern); ret_imgs["horz"] = horz_img

        if disp_cmd == "final":
            return {"final":horz_img}
            # ret_imgs["final"] = horz_img

        return ret_imgs 

    def add_noise(self):
        self.img = noise(self.img, noise_pts=self.img.shape[0]*2, min_pt_sz=self.img.shape[0]/150, max_pt_sz=self.img.shape[0]/50)
        return self.img

    def apply_mask(self):
        self.img = pre_process(self.img, des=["mask"],colors=True)
        return self.img

    def calc_kernel_size(self):

        reduction_coeff = self.params.access("kernel_coeff", universal=True)
        row_wid = self.params.access("avg_bot_width_pxl")
        kernel_sz = round(row_wid*reduction_coeff)

        assert type(kernel_sz) == int, "The kernel size must be an integer"

        self.vert_kern = np.ones((1,kernel_sz),np.uint8)
        self.horz_kern = np.ones((kernel_sz,1),np.uint8)
    
    def calc_img_crop(self, row_width, spacing):
        pass


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
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\python_soybean_c\\Test_imgs\\Special_tests",
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\corn\\imgs",
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\Drone_images\\Winter Wheat",
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\sm_soybeans\\imgs",
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\sm_soybeans\\Quite_weedy"
        ]

    def run(self, func, **kwargs):
        for path in self.test_paths:
            disp.b = False
            self.data = prep('sample', path)["imgs"]
            for img in self.data:
                if disp.b == True: break
                func(img,**kwargs)

    def no_proc(self, img):
        show([img])
    
    def kernel_img(self, img, **kwargs):
        kernel = kerneling()
        imgs = kernel.apply_kernel(img, **kwargs)

        assert type(imgs) == dict, "The returned images must be a dictionary"

        show(imgs)

    
        

if __name__ == "__main__":
    tests = test()
    tests.run(tests.kernel_img, apply_mask=True, apply_noise=False)