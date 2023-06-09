"""
Author: Joel Weber
Title: Prim_detector.py
Overview: This is the primary control module that serves as the initail delegation hub for all the program functions. From here the test data is loaded and passed to the appropriate modules.
Issues: This is definitely an unfinished module and may be prone to unexpectedly crash however work is in progress to establish the detector ideas and then to preform rigourous bug elimaination at a 
later time when the project is more mature. 
"""
# edge_detection
# This is the research program the will test the viability of edge detection upon multipe hieristics
# There will also be some testing for the optimal image preprocessing

import cv2
import numpy as np
import path
import sys,os
from icecream import ic
import time


from inceremental_masks import hough_assessment as hough 
import contour_detect as cont

sys.path.append(os.path.abspath(os.path.join('.')))

from Modules.prep_input import interpetInput as prep 
from Modules.display_frames import display_dec as disp
from Modules.Image_Processor import apply_funcs as pre_process 

from aggregate_lines import ag_lines as agl

from Adaptive_params import Adaptive_parameters as ap
from MAKE_SOME_NOISE import noise 

params = ap.param_manager()
def load_modules(module_list):
    reqs = {"cv2", "np", "time", "ic", }


def edge_detection(src_im, des = ["sobelx","sobely","sobelxy","laplacian"]):

    #Sobel Edge Detection:
    def sobel():
        ret = {}
        # img = pre_process(src_im, des=["blur", "gray"])
        img = pre_process(src_im, des=["kernel"])
        def testKernel():
            ret = {"img":src_im,"Radical":cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=31)}
            for i in range(13, 21, 2):
                ret["sobelx{}".format(i)] = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=i)
            return ret

        if "sobelx" in des:
            ret["sobelx"] = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=15)
        if "sobely" in des:
            ret["sobely"] = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=15)
        if "sobelxy" in des:
            ret["sobelxy"] = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=15)

        return ret

        # return (sobelx,sobely,sobelxy)
    # Laplacian Edge Detection:
    def laplacian():
        img = pre_process(src_im, des=["kernel", "gray"])
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        return laplacian
    sobel_im= sobel()

    ret = sobel()
    # return {'sobelX':sobel_im[0],'laplacian':laplacian()}
    return{'canny':pre_process(src_im,des=['kernel','mask']),'org':pre_process(src_im,des=['kernel'])}

"""
The filters that the image passes through 
 1. Resize to (640,360)
 2. Mask 
 3. Blur 
 4. 
"""
def inc_color_mask(img, upper=[75,255,255], lower = [30,50,50],resolution=10, noise_tst = True,hough_cmd = "avg"):


    no_noise = img.copy()
    if noise_tst:
        img = noise(img) 
    no_noise = pre_process(no_noise, des=["resize","mask"])
    img = pre_process(img, des=["resize"])

    # Initailize the contour class:

    def map():
        iterations = np.array([(upper[0]-lower[0])/resolution,(upper[1]-lower[1])/resolution,(upper[2]-lower[2])/resolution])
        return iterations
    increments = map()
    org = pre_process(img, des=["kernel","mask"],upper=upper,lower=lower,colors=True)
    ret = {"org":img ,"mask_init":org}
    # ret["org"] = img  
    # ret["mask_init"] = org

    # To group lines change the command from "none" to "avg"
    # add "disp_steps" to display the layers of masks to the output
    # imgs = hough(img,org,no_noise,hough_cmd)
    imgs = hough(img,org,no_noise,hough_cmd,"disp_steps")
    up = upper[0]
    low= lower[0]


    def get_masks(org,up,low):
        cur_msk = pre_process(org, des=["blur","mask","kernel"],upper=up,lower=low,colors=True)
        bin_mask = pre_process(org, des=["mask","kernel"],upper=up,lower=low)
        return bin_mask,cur_msk

    total_mask = get_masks(org,upper,lower)
    imgs.slope_filter(total_mask[0],total_mask[1])

    # for i in range(1,resolution+1):
    for v in imgs.color_splits(30,12):

        up = np.array([v[1],upper[1],upper[2]])
        low = np.array([v[0],lower[1],lower[2]])
        bin_mask,cur_msk = get_masks(org,up,low) 

        # Run the blob detections on the image
        # Initalize the contour class:
        def conts():
            grouper = cont.cont_detec(bin_mask)
            grouper.createContours(mode="ext")
            grouper.filterCont(thresh=100)
            return grouper.cont_im
        # Group the lines that are deemed good by slope and  intercept

        imgs.slope_filter(bin_mask,cur_msk)
        # ret["mask{} {}".format(v[0],v[1])] = cur_msk

    # Create a mask for the image
    
    if hough_cmd == "avg":
        total_lns = imgs.good_lns
        res_img = org.copy()

        final_rows = agl(img, total_lns)
        pts, gr_lines = final_rows.calc_pts([0,img.shape[0]])

        res = final_rows.disp_pred_lines(pts)
        # ic(gr_lines)
        imgs.ret["final"] = imgs.group_lns(gr_lines,res_img,grp = False)

    return imgs.ret


def main():


    target = 'Resized_Imgs'
    weeds = 'Weedy_imgs'
    drones = "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Images/Drone_images/Winter Wheat"
    vids = 'Drone_images/Winter_Wheat_vids'
    # ic.disable()
    dataCont = prep('sample',drones)    

    disp(dataCont, inc_color_mask)
    # disp(dataCont, edge_detection)


if __name__ == "__main__":
    main()