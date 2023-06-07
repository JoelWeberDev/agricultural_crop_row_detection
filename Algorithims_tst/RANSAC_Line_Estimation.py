'''
RANSAC_Line_Estimation
Planning: Take a masked image as an input
    1. Gather some data about the image (size, shape, etc)
    2. Use that data to create paramters that will tailor the RANSAC algorithm to the image
        - Min number of inlier samples that will quilfy a line
        - Max number of iterations based on the ratio of green pixels in the frame to the total of the mask and the number of rows in the image.
        - The error margin within the vanishing point for the x coordinate
        - The proximity that will define a point as an inlier: 
         - This is determined by the estimated width of a row due to the pixel calculations
         - In order to do this the most effectively one must make both the outer edges of the line narrow accordingly to the vanishing point
    3. Take these paramters and run the RANSAC algorithm
     - The algorithim will implement the perameters using these following ideas:
      1. Pick an arbirtray point within the x range for the vanishing point
      2. 
'''


import numpy as np
from icecream import ic
import cv2
import os
import sys
from matplotlib import pyplot as plt
import time 


try:
    sys.path.append(os.path.abspath(os.path.join('.')))
    import Modules.Camera_Calib.Camera_Calibration as cam    
except ModuleNotFoundError:
    print("Module not found using an alternative path")
    import Modules.Camera_Calib.Camera_Calibration as cam
try: 
    from inceremental_masks import hough_assessment 
except ModuleNotFoundError:
    raise("The hough_assessment module is not found")
    

class Ransac_lines(object):
    def __init__(self, img) -> None:
        self.cam_access = cam.loaded_cam

        vp_name = "vanishing_point{}x{}".format(img.shape[0],img.shape[1])
        self.vanishing_point= self.cam_access.data[self.cam_access.find_key(vp_name)][vp_name]
        self.vp_calculator = cam.vanishing_point_calculator()

        self.img = img
        self.width = img.shape[1]
        self.height = img.shape[0]

        self.inc_mask = hough_assessment(img)
        self.adp_params = self.inc_mask.adaptive_params

        self.conversion_table = {"m":1000,"mm":1,"cm":10,"km":1000000,"in":25.4,"ft":304.8,"yd":914.4,"mi":1609344, "mm":1}

    # This requeires the image to be a masked sample of a single dimension
    def process_image(self,image):
        size = image.shape
        total_green = np.count_nonzero(image)
        return total_green, size

    def determine_error_margin(self, mask):
        self.cur_pts = np.sum(mask > 0) #** This could potentially be made more efficient by using an exisiting mask iteration somewhere up the chain of processing
        self.cur_ratio = self.cur_pts/self.inc_mask.total_pts
        self.row_base_pxls = (self.cur_ratio*mask.shape[0]) / self.adp_params.access("row_count")
        

    # This is desigend to be the control function of the class and it reqires a range of incremenatlly masked images as well as the original mask 
    def process_mask_range(self,mask,res):
        pass

    # Joel's method for the vanishing point ransac algorithm
    """
    Approach: 
    1. Paramters: Masked image, vainishing point of the image (should be identical for every sample and thus will only need to be calculated once), the inlier hursitic for the ransac, error range for the vanishing point
    2. Calculate the image into a l->r point count along the matrix of the image
    3. Calculate two line that define the edges of the inliers
    4. iterate through the rows of the lr sum matrix and use the x,y values at the line for that row to detrmine where the lines intersect that row.
      - If there is a large gap between rows that have any substanciated number of points disquaify that for for a too large line gap
    5. Take the near and far inlier line calculations and subtract them to get a total number of contributing pixels for that row
    6. Add that total to the sum of the inlier pixels
    7. Repeat this process for all the x-value intervals along the width of the image
    8. Take the best preforming x-value intervals overall that exceed a line threshold and add those to a group of good rows that will be passed along as probable lines that describe the crop rows. 
    """
    def ransac_rows(self,mask, inc=1, thresh = 200):
        # self.mask = mask
        self.mask = self.inc_mask.ret["mask_init"]
        self.lr_mat = self.lr_mat()
        pt_vals = np.array([[self.inlier_huristic(i),i] for i in np.arange(0,self.width,inc)])
        filt_cond = pt_vals[:,0] > thresh
        # return pt_vals[filt_cond]
        pt_arr = pt_vals[filt_cond]
        # ic(pt_arr)
        return {"mask": self.ln_img(pt_arr, self.mask), "other":self.img, "test":self.tst_vp(self.img)}

    def ln_img(self,pts,img):
        vp = self.vanishing_point
        ic(vp)
        for pt in pts:
            cv2.line(img,(int(pt[1]),720),(int(vp[0]),int(vp[1])),(255,255,255),2)
        return img

    def tst_vp(self, img):
        for i in np.arange(0,self.width,96):
            cv2.line(img,(int(i),720),(int(640),int(self.vanishing_point[1])),(255,255,255),2)
        return img

    def inlier_huristic(self,x_val):
        """
        This is based off the number of positive points within the masked image and the ratio to the total number of points in the loose mask
        The variance can be a value form 0 to 1 that is a percentage of the image width that will represent the distance of one huristic line from the line center
        """
        err_marg= int(self.adp_params.access("max_err")*self.width)
        lines = [self.vp_calculator.cartesian_lns(np.array([self.vanishing_point,[x_val+(i*err_marg),self.height]])) for i in range(-1,2,2)]

        total = 0
        for i in np.arange(0,self.mask.shape[0],1):
            total+=self.solve_row_line(lines,i)

        return total

    def solve_row_line(self,lines,row_num):
        min_max = np.array([(row_num-ln[1])/ln[0] if (row_num-ln[1])/ln[0] >= 0 else 0 for ln in lines])
        row_pts = abs(self.lr_mat[row_num, int(min_max[0])] - self.lr_mat[row_num, int(min_max[1])])
        return row_pts
            
    def lr_sum(self):
        sum  = 0
        def inc_row(val):
            nonlocal sum
            try:
                if val > 0:
                    sum +=1
            except Exception as e:
                ic(val)
                raise e
            return sum

        def next_row(row):
            nonlocal sum
            sum = 0
            return np.fromiter(map(inc_row, row), dtype=int)
        
        lr_mat = next_row(self.mask[0])
        lr_mat = np.array([next_row(row) for row in self.mask])
        return lr_mat

    def lr_mask(self):
        bin_im = np.where(self.mask > 0, 1, 0)
        return np.cumsum(bin_im,axis=1)


def test():
    
    from Modules.prep_input import interpetInput as prep
    from Modules.display_frames import display_dec as disp


    target = 'Resized_Imgs'
    weeds = 'Weedy_imgs'
    drones = "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Images/Drone_images/Winter Wheat"
    dataCont = prep('sample',drones)

    model = Ransac_lines(dataCont["imgs"][0])

    ic(model.ransac_rows(model.inc_mask.ret["mask_init"]))
    model.mask = model.inc_mask.ret["mask_init"]

    start = time.time()
    
    model.lr_mask()
    end = time.time()
    # printing the execution time in secs.
    # nearest to 3-decimal places
    ms = (end-start) 
    print(f"Elapsed {ms:.03f} secs.") 

    disp(dataCont, model.ransac_rows)

if __name__ == "__main__":
    test()