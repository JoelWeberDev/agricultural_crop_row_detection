# Watershed_algorithm

import cv2
import numpy as np
import matplotlib.pyplot as plt

import path
import sys,os

sys.path.append(os.path.abspath(os.path.join('.')))

from Modules import Image_Processor as ip
from Modules.prep_input import interpetInput as prep 
from Modules.display_frames import display_dec as disp


# Function reflection:
# This works quite well on smaller less significant however once overalp occurs it is not as effective
# Perahaps this could be allievated by using countours and edge canny
def watershed_algorithm(img):
    # preprocessing: We want to obtain the unknown section of the image
    # Create a binarized iteration of the image to invert it in our case this is the mask
    looseMask = ip.procImgs(img,'mask')(np.array([95, 255, 255]), np.array([30, 50, 50]))

    # remove touching masks
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(looseMask, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(mask,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(mask,cv2.DIST_L2,5)
    # Tune the dist_trasnform and it may need to be adaptivley tuned to opimize results for differing conditions
    ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    
    mask = np.zeros_like(img)
    mask[markers == -1] = [0,255,255]


    return ({'shed':mask,'img':img})

def main():

    target = 'Resized_Imgs'
    weeds = 'Weedy_imgs'
    dataCont = prep('sample',weeds)    

    disp(dataCont, watershed_algorithm)

if __name__ == "__main__":
    import system_operations as sys_op
    sys_op.system_reset()
    main()
