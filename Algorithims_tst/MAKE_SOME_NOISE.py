"""
Author: Joel Weber
Title: MAKE_SOME_NOISE
Overview: This module will add noise to the image
Usage: Allows for testing with randomized green noise in the image
"""
# MAKE_SOME_NOISE


import random
import cv2
import numpy as np
from icecream import ic


# The size paramters must be odd values since the points will define a center point
def noise(img, noise_pts= 3000, min_pt_sz = 1 , max_pt_sz = 11):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upper=np.array([75,255,255])
    lower = np.array([30,50,50])
    # ic(img.shape)

    if min_pt_sz%2 == 0:
        min_pt_sz+=1
    if max_pt_sz%2 == 0:
        max_pt_sz+=1

    for i in np.arange(noise_pts):
        sz = random.randint((min_pt_sz+1)/2, (max_pt_sz+1)/2)-1
        col = np.array([random.randint(lower[0], upper[0]), random.randint(lower[1], upper[1]), random.randint(lower[2], upper[2])])
        pt = np.array([random.randint(sz, img.shape[0]-sz), random.randint(sz, img.shape[1]-sz)]) 
        img[pt[0]-sz:pt[0]+sz, pt[1]-sz:pt[1]+sz] = col

    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def eliminate_noise(img):
    return cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

def test(img): 
    # Return a dictionary with the images that should be displayed
    ns = noise(img)
    ret = {"org": img, "noise": ns, "denoise_art": eliminate_noise(ns), "denoise_real": eliminate_noise(img) }

    return ret




if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('.')))
    from Modules.prep_input import interpetInput as prep
    from Modules.display_frames import display_dec as disp

    target = 'Resized_Imgs'
    weeds = 'Weedy_imgs'
    drones = "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Images/Drone_images/Winter Wheat"
    vids = 'Drone_images/Winter_Wheat_vids'
    dataCont = prep('sample',vids)    


    disp(dataCont, test)

