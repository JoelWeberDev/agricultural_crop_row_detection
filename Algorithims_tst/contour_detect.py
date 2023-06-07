"""
Author: Joel Weber
Title: Detect Contours
Overview: This was used to detemine if the opencv contours were a promising development to detect rows from their edges
Notes: The contour idea was not super effective therfore the this file may not be used
"""
# contour_detect

# Path: Modules\contour_detect.py
# Module usitily: Extract the contours from a binarized image. 
#  Useful filtering functions for the contours are also included:
# - filterCont: Filter the contours by a threshold value
# - averageContour: Average the contours to reduce the number of points
# - catConts: Categorize the contours into a list of contours for each template


import cv2
import numpy as np

class cont_detec(object):
    def __init__(self,org) -> None:
        self.im = org
        self.cont_im = np.zeros(self.im.shape,dtype= np.uint8)
        self.im_count = 0
        self.createContours()

    def filterCont(self,thresh=2,average=False,categorize=False):
        filtered = []
        
        for cont in self.contours:
            if cv2.contourArea(cont) > thresh: 
                filtered.append(cont)
        if average:
            self.contours = self.averageContour(filtered)
        elif categorize:
            self.contours = self.catConts(filtered) 
        else:
            self.contours = filtered


    def kernelConts(self,k=4):
        self.kerneled = cv2.dilate(self.img,np.ones((k,k),np.uint32)/k**2,iterations = 1) 

    def createContours(self,mode='ext',method='approx', draw_conts=True,**fArgs):
        canny = cv2.Canny(self.im,100,200)
        methodMap={'approx':cv2.CHAIN_APPROX_SIMPLE,'none':cv2.CHAIN_APPROX_NONE}
        modeMap={'ext':cv2.RETR_EXTERNAL,'list':cv2.RETR_LIST,'tree':cv2.RETR_TREE} 

        self.contours,self.heiarchy = cv2.findContours(canny,modeMap[mode],methodMap[method])

        # Apply conoturs to the source image

        cv2.drawContours(self.cont_im ,self.contours,-1,(255,255,255),1)

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

    def catConts(self,contours):
        templates = []
        image = self.im.copy()
        for cont in contours:

            mx = np.amax(cont,axis=0)
            mn = np.amin(cont,axis=0)
            pts = [(mn[0][0],mn[0][1]),(mx[0][0],mx[0][1])]
            # image = cv2.rectangle(image,(pts[2],pts[0]),(pts[3],pts[1]), (0,0,255), 3);            
            image = cv2.rectangle(image,pts[0],pts[1], (0,0,255), 3);            
            templates.append(self.im[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]])
        return image
    