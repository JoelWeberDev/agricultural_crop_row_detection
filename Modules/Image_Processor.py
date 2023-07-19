# Image_Processor
# Module Outline:
# IDEAS: New methods of processing the image and their implementation
# - Preform blob detection on a masked image to identify plants
#   - Weed_Plant filter:
#     - Preform camera calculations to determine approxomately the position and the spacing of the plants
#     Identifying and processing probable rows:
#     - Generate a ranking system by which the rows can be judged (Run all the tests in parallel and don't discard 
#      any rows until the perlimnary tests are completed) 
#      - Use the hough line transform and image masking to identify potential rows
#       - Use the image size presuppositons to identify some probalble rows       
#       - Create a pattern recognition model to identify any unifomity of spacing on potential rows  
#       - Number of lines that are present within a proximity
#       - IDEA: Generate a grayscale mapping of green hue that exist within the boundaries but display their brightnesses
#           - Then use that mapping to orthogonalize the plants regardless if there is overlap  
#   - Then use the hough line on the blobs to allow for less requred processing and identification on the rows
# - Make a green mask of the image

# Path: Modules\Image_Processor.py

#  Session goals:
#  Pattern recognition model:
#    IDEAS:
#     - Use contour detection with a blur to break the model down into leaves and stems
#       - Merge smaller contours into larger ones
#       - Use template matching of areas where green contours are present Template needs to be quite simialr to the object that is the target
#       - Gather quite weedy samples and use them to identify where it falls apart
#       -  

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sklearn as sk

import sys, os

sys.path.append(os.path.abspath(os.path.join('.')))
from Adaptive_params.Adaptive_parameters import param_manager  

ap = param_manager()

# Green mask creator: 
# - Masks all values that resisde within the green hue range
# - generates a grayscale showing the brightness of the green hue

def apply_funcs(img, des=["gray"],**kwargs):
    def mask(img=img):
        if "upper" in kwargs and "lower" in kwargs:
            upper = kwargs["upper"]
            lower = kwargs["lower"]
        else:
            upper = np.array([85, 255, 255])
            lower = np.array([30, 50, 50])
        loosemask = procImgs(img,'mask')(upper,lower)
        maskGrad = procImgs(img,'res')(loosemask)
        # use contrast enahncing filters to generate a better canny results
        # Fixed hist equalization
        # enh = cv2.equalizeHist(maskGrad)
        if "colors" in kwargs and kwargs["colors"]:
            return maskGrad
        return loosemask

    # Convert to grayscale
    def gray(img=img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    def blur(img=img):
        return cv2.GaussianBlur(img, (3, 3), 0)
    # # Apply kernel 

    # Apply Canny edge detection
    def canny(img=img):
        return cv2.Canny(img, 150, 200)

    def resize(img=img):
        return cv2.resize(img, tuple(ap.access("im_dims")))
        

   
    funcs = {"gray":gray, "blur":blur, "canny":canny,"mask":mask,"kernel": procImgs(img,'kernel'),"resize":resize}
    param = img
    for d in des:
        if d in funcs:
            param = funcs[d](param)
    return param

def procImgs(img=None, reqFunc = None, reqIms =['shadeMask']):


    def hsv(img=img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def mask(upper_green = [86, 255, 255], lower_green = [36, 60, 60]):
        if type(upper_green) == type([]):
            upper_green = np.array(upper_green)
        if type(lower_green) == type([]):
            lower_green = np.array(lower_green)

        return cv2.inRange(hsv(), lower_green, upper_green)

    def res(mask=mask(),img=img):
        return cv2.bitwise_and(img, img, mask=mask)

    def binIm():
        return cv2.dilate(res(), np.ones((5,5),np.float32)/25)
    
    def hues(mask=mask()):
        return cv2.cvtColor(cv2.cvtColor(res(mask,hsv()),cv2.COLOR_HSV2BGR),cv2.COLOR_BGR2GRAY)

    # Feed hsv image 
    def multiShadeMask(img=img):
        # print(img.shape)
        # print(img[0,0,0])
        # hue deviation map:
        h,s,v = cv2.split(img)
        # create merge
        merge = cv2.merge([np.absolute(h-60),np.absolute(s-50),np.absolute(v-50)])
        return [merge[:,:,0],merge[:,:,1],merge[:,:,2]]

   # Overarching pattern recognition model:
    def corners(img=img):
        # - Attempt to determine any continuties or reccuruning patterns in the image
        #   - EG: There are many pointed green leaves that appear to be in linear proximitiy of each other and extend the entire screen
        #   - Consider the entire image and not only the masked componnets
        #   - Use the C Haris key featrues function to identify the corners of the image and thus detect patterns
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerHarris(mask(),5,5,0.2)
        cornerImg = cv2.dilate(corners,None)
        img[cornerImg>0.05*cornerImg.max()]=[0,0,255]


        # Detect the edges by using the canny funcition and sharp contrast within the image
        return img 

    def blobDetect():
        values = cv2.SimpleBlobDetector_Params()
        
        # Area parameters
        values.filterByArea = True
        values.minArea = 10

        # Circularity parameters
        values.filterByCircularity = True
        values.minCircularity = 0.3

        # Convexity parameters
        values.filterByConvexity = True
        values.minConvexity = 0.2

        # Inertia parameters
        values.filterByInertia = True
        values.minInertiaRatio = 0.01

        detector =  cv2.SimpleBlobDetector_create(values)

        tgtIm = mask()
        kp = detector.detect(img)
        print(kp)
        return cv2.drawKeypoints(tgtIm, kp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Round to nearest odd number
    # This is used as part of the kernel franchise
    def round_odd(num):
        rnded  = int(num)
        if rnded % 2 == 0:
            odd_range = [rnded-1, rnded+1]
            return min(odd_range, key=lambda x:abs(x-num))
        return rnded

    def kernel(img=img, kernel_sz = None):

        # determine kernel size relative to image size
        if kernel_sz == None:
            k_horz = ap.access("vert_factor")
            k_vert = ap.access("vert_factor")

            # kernel_factor = ap.access("kernel_factor")
            # print(kernel_factor)
            kernel_sz = (round_odd(img.shape[1]/k_horz), round_odd(img.shape[0]/k_vert))
        

    
        kern = np.ones(kernel_sz,np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kern)

    # Dictionary that defines the possible image processing that can be returned
    retIms = {'mask':mask,'hues':hues,'binIm':binIm,'res':res,'hsv':hsv,'shadeMask':multiShadeMask,'corners':corners
    ,'blob':blobDetect,'kernel':kernel}
    ret = {}


    if reqFunc != None:
        try:
            return retIms[reqFunc]
        except KeyError:
            raise Exception('Invalid function request')

    for req in reqIms:
        if type(req) != str: raise Exception('Please enter a valid string')

        elif req in retIms.keys():
            imgs = retIms[req]()
            if type(imgs) == list:
                for i in imgs:
                    ret[req] = i
            else:
                ret[req] = imgs
        else: 
            raise Exception('Invalid image request try {}'.format(retIms.keys()))
    return ret

# This is the underlying model that will recgnize leave patterns and identify the rows based on the aggragation of specifc leaf types
def identLeaves(img):
    looseMask = procImgs(img,'mask')(np.array([95, 255, 255]), np.array([30, 50, 50]))
    maskGrad = procImgs(img,'hues')(looseMask)
    # use contrast enahncing filters to generate a better canny results
    # Fixed hist equalization
    enh = cv2.equalizeHist(maskGrad)
    # Adaptive hist equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
    # adpEnh = clahe.apply(maskGrad)
    

    canny  = cv2.Canny(enh,350,400) 
    # Filter the contours by their area and eliminate the ones that fall below a specific threshold
    # If the contours are under the area threshold group them with adjacent contours until the abide within the threshold
    def filterCont(contours,kwargs={}):
        filtered = []
        thresh = 2 if 'areaThresh' not in kwargs else kwargs['areaThresh']
        # print(thresh)
        
        for cont in contours:
            if cv2.contourArea(cont) > thresh: 
                filtered.append(cont)
        if ('average' in kwargs and kwargs['average']):
            return averageContour(filtered) 
        elif ('categorize' in kwargs and kwargs['categorize']):
            print('categorize')
            return catConts(filtered) 
        return filtered


    def kernelConts(img=canny,k=4):
        kernel = cv2.dilate(img,np.ones((k,k),np.uint32)/k**2,iterations = 1) 
        return kernel 

    def createContours(srcIm=canny,mode='ext',method='approx',func=None, draw_conts=True,**fArgs):
        methodMap={'approx':cv2.CHAIN_APPROX_SIMPLE,'none':cv2.CHAIN_APPROX_NONE}
        modeMap={'ext':cv2.RETR_EXTERNAL,'list':cv2.RETR_LIST,'tree':cv2.RETR_TREE} 

        contours,heiarchy = cv2.findContours(srcIm,modeMap[mode],methodMap[method])

        # Apply conoturs to the source image

        if(draw_conts):
            cv2.drawContours(ret := np.zeros(img.shape,dtype= np.uint8),func(contours,fArgs) if len(fArgs) != 0 else func(contours) if func != None else contours,-1,(0,0,255),1)
        else: 
            ret = func(contours,fArgs) if len(fArgs) != 0 else func(contours) if func != None else contours
        return ret

    # Determine the range of x-values that the contours exist within
    # For each x-parallel that the contour lies on average the x-values
    # 
    def averageContour(contours):
        ret = []
        
        for cont in contours:
            ysort = cont[cont[:,0,1].argsort()]
            for i in range(len(ysort)):
                imatch = i
                sum = 0
                while imatch < len(ysort): 
                    if ysort[i][0][1] != ysort[imatch][0][1] :
                        break
                    sum += ysort[imatch][0][0]
                    imatch += 1
                for j in range(i,imatch):
                    ysort[j][0][0] = int(sum/(imatch-i))
                i = imatch
            ysort = np.unique(ysort,axis=0)
            ret.append(ysort)
        return ret 

    def catConts(contours):
        templates = []
        image = img.copy()
        for cont in contours:

            mx = np.amax(cont,axis=0)
            mn = np.amin(cont,axis=0)
            pts = [(mn[0][0],mn[0][1]),(mx[0][0],mx[0][1])]
            # image = cv2.rectangle(image,(pts[2],pts[0]),(pts[3],pts[1]), (0,0,255), 3);            
            image = cv2.rectangle(image,pts[0],pts[1], (0,0,255), 3);            
            templates.append(procImgs(img,'res')(looseMask)[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]])
        return image
            # templates.append((img)[np.min(y := pts[0]):np.max(y),np.min(x := pts[1]):np.max(x)])
        # import feature_match as fm
        # print(len(templates))

        # cluster templates with 


    # external = createContours(kernelConts(),'ext','approx')
    # high_res = createContours(canny,mode='tree',method='approx',func=filterCont,categorize=True)
    # tmps_match = createContours(canny,mode='tree',method='approx',func=filterCont,draw_conts=False,categorize=True,areaThresh=20)
    # Heiarchy array format: [Next, Previous, First_Child, Parent]
    # return {'External':external,'High Res':high_res} 
    
    # return {"org":img,"loose" : canny,"low":cv2.Canny(enh,150,200,3,True),"high":cv2.Canny(enh,450,500,3,True),"tight":cv2.Canny(enh,450,475,3,True)}
    return {"org":img,"loose" : canny,"low":cv2.Canny(enh,350,700,21,L2gradient=True)}
     

# Row ranking system:
def rowRanking(img,funcs = [None]):
    # Funtions: 
    #  - Camera calculations and image dimensions
    #  - Hough Line Transform
    #  - Brightness gradient
    #  - Pattern recognition: Search for uniformity in the spacing of the plants
    #   - Contrast atleast two rows and determine the ones that have the most similar traits then give the a favourable ranking in the model
    pass
    
### TEST CASES ###
def test():
    import prep_input as prep
    import display_frames as disp
    target = 'Resized_Imgs'
    weeds = 'Weedy_imgs'
    dataCont = prep.interpetInput('sample',weeds)    
    def testGreenMask():
        disp.display_dec(dataCont, procImgs)
    # testGreenMask()
    def testPatternRecog():
        disp.display_dec(dataCont,identLeaves)
    testPatternRecog()

if __name__ == '__main__':
    import system_operations as sys_op
    sys_op.system_reset()

    test()