"""
Author: Joel Weber
Date: 05/30/2023
Description: Here each individual frame is processed primarily by color through the divison of the frame into diffenent green hues and a hough line detector applied to each of these masks. 
After this all the lines are combined and filtered in aggregate_lines.py to determine the most likely lines that represent the rows.
"""
# inceremental_masks

# Next steps: Build a blob detector that is able to groups and also parse the smaller areass of noise
#  - This will allow us to extract only the strong detections of the certain color hues and ignore the noise
#  - We can then also use the hough to create one broader stroke rather than many that waste computation and need to be parsed and 
#    avearaged.
import cv2
import numpy as np
from icecream import ic
import sys,os
import math

from Group_lines import group_lns as grp


sys.path.append(os.path.abspath(os.path.join('.')))

import Modules.Camera_Calib.applied_calculations as calcs
from Modules.Image_Processor import procImgs as proc
from Modules import line_calculations as lc

from Adaptive_params import Adaptive_parameters as ap
adp_vals = ap.value_calculations(efficiency=True)

# from Modules import display_frames as disp

class hough_assessment(object):
    def __init__(self,org,mask_init=None,pure=None,*args) -> None:

        if mask_init is None:
            mask_init = proc(org,'mask')(np.array([85,255,255]),np.array([30,50,50]))
        if pure is None:
            pure = org

        self.args = args

        self.ret = {"org":org, "pure":pure,"mask_init":mask_init}
        
        # self.cmd = cmd

        self.im_count = 0
        self.total_pts = np.sum(mask_init >0)

        self.good_lns = []
        self.grouper = grp()
        self.calc_appl = calcs.generic(org.shape)

        self.im_size = self.ret['org'].shape

        self.adaptive_params = ap.param_manager()
        self.row_num = self.adaptive_params.access("row_num")
        self.err_green = self.adaptive_params.access("err_green")
        self.sense = self.adaptive_params.access("sensitivity")

    def appy_ln(self,mask,probibalistic=True):
        import time 

        # A method to use the conventional method was also developed however the time and complextiy that it requires far exceeds that of the probibalistic method. The only way in which the simple HoughLines could 
        # be superior is if the theta values were made very rigid
        params = adp_vals.determine_hough(np.sum(mask> 0))
        # params = self.get_params(mask)

        if not probibalistic:
            start = time.time()
            thetas = self.get_theta_bounds()
            lines = cv2.HoughLines(mask, 1, np.pi / 120, params[0], thetas[0], thetas[1])
            end = time.time()
            ic(end-start,thetas)
            return self.decode_ln_pts(lines)

        start = time.time()
        lines = cv2.HoughLinesP(mask,1,np.pi/120,params[0],minLineLength=params[1],maxLineGap=params[2])
        end = time.time()

        # ic(end-start)
        ic(params)
        # lines = cv2.HoughLinesP(mask,1,np.pi/180,50,minLineLength=100,maxLineGap=200)
        return lines

    def _calc_theta_range(self, x,y):
        thetas = np.array([lc.slope_to_theta(s) for s in self.calc_appl.estimate_slope(np.array([x,y]), ret_range=True)])
        np.sort(thetas)
        return thetas

    def get_theta_bounds(self):
        edge_pts = np.array([[0,self.im_size[0]],[self.im_size[1],self.im_size[0]]])
        thetas = np.array([self._calc_theta_range(x,y) for x,y in edge_pts])
        edge_thetas = np.array([np.min(thetas[:,0]),np.max(thetas[:,1])])
        return edge_thetas 

        

    def decode_ln_pts(self,lines):
        ret_lns = []
        for r_theta in lines:
            r,theta = r_theta[0]
            a = np.cos(theta)   
            b = np.sin(theta)
            x0 = a*r
            y0 = b*r
            ret_lns.append(np.array([[int(x0 + 1000*(-b)), int(y0 + 1000*(a)),int(x0 - 1000*(-b)), int(y0 - 1000*(a))],[]]))
        # ic(ret_lns) 
        return ret_lns 
            

    # Method outline: This should assess the size of the rows determined from the user inputs and the camera parameters to determine approxomately how many points should consitute a row
    #  * We could also use this to determine the size of kernel that is fitting to properly group blobs
    # def get_params(self,cur_mask):
    #     # self.cam_calcs(cur_mask)
    #     # These values change based on the input of the mask

    #     # self.cur_pts = np.sum(cur_mask > 0) #** This could potentially be made more efficient by using an exisiting mask iteration somewhere up the chain of processing
    #     # self.cur_ratio = self.cur_pts/self.total_pts
    #     # self.row_base_pxls = (self.cur_ratio*self.im_size[0]) / row_num
    #     # vals = [self._row_pixels(),self._line_length(),self._line_gap()]

    #     return vals

    # How to estimate the amount of pixels that should consitiute a line with the given values:
    #  - The line cannot be below a minimum number of pixels

    def _row_pixels(self):
        # consants: Values that set the bounds of the calculations for a row
        # Perhaps this should be a plastic value such that it derives from the number of pixels in the image. The most significant issue with this is that the lower bound may never be reached depending on how it is
        # calculated.
        # pixel_min = int(self.adaptive_params.access("min_pixels_fact")*self.cur_pts)

        # pixel_min = int(math.log(self.cur_pts,1.5)) if self.cur_pts > 100 else 2
        pixel_min = int(1.5*math.log((self.cur_pts*self.err_green/self.row_num),1.3)*(1-(self.sense/2))) if self.cur_pts > 300 else 3
        # ic(pixel_min,self.cur_pts)

        # Variable values
        adj_factor = self.adaptive_params.access("pixels_per_row_percent")  

        # Calculations
        # Three desired value: Min line points, min line length, max line gap 

        # Min line points
        # Calculation method: The number of points within the image is divided be the number of rows. Then there is an adjustment factor that is wieghted by the ratio of points to the total points in the image
        # How is the adjustment factor determined? 
        #  - The ratio is found and then multiplied by a wieghting constant that may need to be tuned, but it errs on the side of less effect. 

        self.line_pxls = self.row_base_pxls + (self.row_base_pxls* self.cur_ratio * adj_factor) #* This is the adjustment factor and the precentage may need to be adjusted and eventially adaptive
        if self.line_pxls < pixel_min:
            return(pixel_min)
        return(int(self.line_pxls))
        # return pixel_min
        
    # This is a somewhat private method and is typicall called from the get_params method 
    def _line_gap(self):
        max_gap_bound = self.adaptive_params.access("max_line_gap_ratio") * self.im_size[0] 
        min_gap_bound = self.adaptive_params.access("min_line_gap_ratio") * self.im_size[0]
        max_gap = min_gap_bound + (1-self.cur_ratio)*max_gap_bound
        return(int(max_gap))

    def _line_length(self):
        min_ln_bound = self.adaptive_params.access("min_line_length_ratio") * self.im_size[0]
        min_ln_len = min_ln_bound + (self.cur_ratio)*min_ln_bound
        return(int(min_ln_len))


    def slope_filter(self,mask,res=None):
        # Create a numpy array by shape
        if res is None:
            res = np.zeros_like(self.ret["org"])

        good_lines = []
        lines = self.appy_ln(mask)
        err_margin = 30 #* This is the error margin for the x_point offset from the vanishing point
        if lines is None:
            self.ret["mask {}".format(self.im_count)] = res
            self.im_count += 1
            return

        for ln in lines:

            x1,y1,x2,y2 = ln[0]

            try:
                slope = (y2-y1)/(x2-x1)
            except ZeroDivisionError or RuntimeWarning:
                slope = np.inf

            self.slope_color(10)

            if "none" in self.args:
                
                # if abs(1/slope) < 1.7: #replace with cam calcs values
                cv2.line(res, (x1,y1), (x2,y2), self.ln_col, 2)
                continue

            # print(1/slope, ideal_slope)
            # Invert the slopes since the infinite slope is the base line

            # if True:
            # ** Use the adaptive parameters to input the error margin for the slope also
            if self.calc_appl.estimate_slope(np.array([x1,y1]), slope = slope):
                
            # if 1/slope > ideal_slope[0] and 1/slope < ideal_slope[1]: #replace with cam calcs values
            # if slope > ideal_slope[0] and slope < ideal_slope[1]: #replace with cam calcs values
                """** special Note: We use the x values at the lowest point of the image for that is the place where we want to process that rows with the most certainty"""
                try:
                    intercept_bot = int((mask.shape[0]- (y1-(slope*x1)))/slope)
                except ZeroDivisionError as e:
                    if x1 > mask.shape[1]/2-err_margin and x1 < mask.shape[1]/2+err_margin:
                        intercept_bot = mask.shape[1]/2
                        slope = 1e10
                    else: 
                        continue
                
                except OverflowError as e:
                    if x1 > mask.shape[1]/2-err_margin and x1 < mask.shape[1]/2+err_margin:
                        intercept_bot = mask.shape[1]/2
                        slope = 1e10
                    else: 
                        continue

                except ValueError or RuntimeWarning:
                    continue
                if "avg" in self.args:
                    good_lines.append([slope,intercept_bot])
                    self.good_lns.append([slope,intercept_bot])
                else:
                    cv2.line(res, (x1,y1), (x2,y2), self.ln_col, 2)

        ret_im = res
        if "avg" in self.args:
            if len(good_lines) == 0:
                return
            ret_img = self.group_lns(np.array(good_lines), res)
            # self.ret["mask {}".format(self.im_count)] = self.group_lns(np.array(good_lines), res) 
        if "disp_steps" in self.args:
            # self.ret["mask {}".format(self.im_count)] = rs
            self.ret["mask {}".format(self.im_count)] = ret_im 
        self.im_count += 1

    def group_lns(self,lines,res=None,ln_tp = "avg", grp = True):
        # Gorups the lines and put them on the image with each group being a different color
        if grp:
            lines = self.grouper.groupLines(lines)

        # #** To make more efficinet convert the lines into points at the grouping phase
        for i,lns in enumerate(lines):
            self.slope_color(i,len(lines),0)
            # Average the lines and then place the avearged ones on the image
            if ln_tp == "avg":
                if len(lns) == 0:
                    continue
                elif len(lns) == 1:
                    pts = self.get_pts(lns[0][0],lns[0][1])
                else:
                    av_ln = self.avg_lns(np.array(lns))
                    # if av_ln[1] > 300 and av_ln[1] < 350 and not grp:
                    #     ic(lns)

                    try:
                        pts = self.get_pts(av_ln[0],av_ln[1])
                    except IndexError:
                        continue

                cv2.line(res, (pts[0],pts[1]), (pts[2], pts[3]), self.ln_col, 2)
            
            # Place all the grouped lines on the image
            else: 
                for ln in lns: 
                    pts = self.get_pts(ln[0],ln[1])
                    cv2.line(res, (pts[0],pts[1]), (pts[2], pts[3]), self.ln_col, 2)
        return res
        
        
    def avg_lns(self, pts_gr=None): 
        # Check if all the slopes have the same sign
        if (pts_gr[:,0] < 0).all() or (pts_gr[:,0] > 0).all():
            # If they do then average the slopes and the intercepts
            return np.average(pts_gr,axis=0)
        else:
            sl_inv = 1/pts_gr[:,0]
            # ic(sl_inv)
            return np.array([1/np.average(sl_inv), np.average(pts_gr[:,1])])


    # To obtain optimal results ensure that samp_num > (upper-lower)/col_inc
    def color_splits(self, samp_num, col_inc, upper= 65,lower = 30):
        # map values between upper and lower such that there is overlap between the values
        color_map = np.ndarray((samp_num,2))
        incs = ((upper-col_inc)-lower)/(samp_num-1)
        for i in range(samp_num):
            color_map[i] = [int(lower+(incs*i)), int((lower+(incs*i))+col_inc)]
        return color_map
    

    def get_pts(self,slope, intercept):
        try:
            int_nrm =round(-1*(intercept*slope-self.ret["org"].shape[0])) 
        except OverflowError :
            int_nrm = 10e10
        x1 = -1*int(int_nrm/slope)
        y1 = 0
        x2 = int(intercept)
        y2 = self.ret["org"].shape[0]
        return (x1,y1,x2,y2)
    
    def slope_color(self,slope, up_sl = 10, low_sl = 0.75 ):
        # This function will take the slope of a line and return a color value based on the slope
        # This will be used to color the lines based on their slope

        # This is the color map that will be used to color the lines based on their slope

        # Alternaive to this could be the random color generator
        ln_col = np.random.randint(50,255,(3),dtype=int)
        self.ln_col = [ln_col.item(0),ln_col.item(1),ln_col.item(2)]
        

        # self.ln_col = [0,0,0]
        # upper = 255*3
        # lower = 127
        # up_sl = 10
        # low_sl = 0.75
        # col_inc = (upper-lower)/(up_sl-low_sl)
        # sl_col = int(abs(slope)*col_inc)
        # for i in range(3):
        #     if sl_col > 255:
        #         sl_col -= 255
        #         self.ln_col[i] = 255
        #     else:
        #         self.ln_col[i] = sl_col

    def pre_process(self, img, des=["gray"],**kwargs):
        def mask(img=img):
            if "upper" in kwargs and "lower" in kwargs:
                upper = kwargs["upper"]
                lower = kwargs["lower"]
            else:
                upper = np.array([85, 255, 255])
                lower = np.array([30, 50, 50])
            loosemask = proc(img,'mask')(upper,lower)
            maskGrad = proc(img,'res')(loosemask)
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
            

    
        funcs = {"gray":gray, "blur":blur, "canny":canny,"mask":mask,"kernel":proc(img,'kernel')}
        param = img
        for d in des:
            if d in funcs:
                param = funcs[d](param)
        return param


    def inc_color_mask(self, img, appl_fxn = None, upper=[75,255,255], lower = [30,50,50],resolution=7):
        # Initailize the contour class:

        def map():
            iterations = np.array([(upper[0]-lower[0])/resolution,(upper[1]-lower[1])/resolution,(upper[2]-lower[2])/resolution])
            return iterations
        increments = map()
        org_mask= self.pre_process(img, des=["mask","kernel"],upper=upper,lower=lower,colors=True)
        self.ret = {"org":img ,"mask_init":org_mask}

        up = upper[0]
        low= lower[0]

        def get_masks(org,up,low):
            cur_msk = self.pre_process(org, des=["blur","mask","kernel"],upper=up,lower=low,colors=True)
            bin_mask = self.pre_process(org, des=["mask","kernel"],upper=up,lower=low)
            return bin_mask,cur_msk

        total_mask = get_masks(org_mask,upper,lower)
        self.slope_filter(total_mask[0],total_mask[1])

        for v in self.color_splits(7,10):

            up = np.array([v[1],upper[1],upper[2]])
            low = np.array([v[0],lower[1],lower[2]])
            bin_mask,cur_msk = get_masks(org_mask,up,low)

            if appl_fxn != None:
                bin_mask = appl_fxn(bin_mask,cur_msk)
            else:
                self.slope_filter(bin_mask,cur_msk)

        return self.ret


def hough_test(img):
    from Modules.Image_Processor import apply_funcs as pre_process 

    img = pre_process(img, des=["resize"])
    org = pre_process(img, des=["kernel","mask"],colors=False)

    ha = hough_assessment(img,org)
    mask = ha.ret["mask_init"]
    lns = ha.appy_ln(mask)

    if lns is not None:
        for ln in lns:
            x1,y1,x2,y2 = ln[0]
            cv2.line(mask, (x1,y1), (x2,y2), (255,255,0), 2)

    return {"mask":mask}
    

def main():
    from Modules.prep_input import interpetInput as prep 
    from Modules.display_frames import display_dec as disp
    drones = "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Images/Drone_images/Winter Wheat"
    vids = 'Drone_images/Winter_Wheat_vids'
    # ic.disable()
    dataCont = prep('sample',drones)    

    disp(dataCont, hough_test)

if __name__ == "__main__":
    main()