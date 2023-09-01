"""
Author: Joel Weber
Date: 15/07/2023
Title: Proximal search
Description: Analyses the green points around a line to determine if it is actually in the center of a row also could catch if the line is not actually representative of a row

Input Data:
    - Set of averaged lines 
    - The width of the row

Strategy:
    1. Iterate through each averaged line and check the area around for the following conditions: Total green points, green point density map, 
    2. For the width+ an error margin create a cropped image
    3. Apply a fairly conservative horizontal kernel to group the green areas around the central areas
    4. Perform a cumulative sum on the cropped area of the image.
    5. Find the center of the cum sum matrix for each row.
    6. Determine the line of best fit for all the centers of each row
"""

import cv2
import numpy as np
import sys,os
import math 

sys.path.append(os.path.abspath(os.path.join('.')))
from Adaptive_params import Adaptive_parameters as ap

class proximal_search(object):
    def __init__(self):
        pass

"""
Descrption: We need to make a crop around the line with the width of the row on either side. However the dillema is that the lines will be diagaonal and matrixies don't especaill appreciate that.
We therefore need to calculate the two boundary lines and make an array of zeros the height of the image and then the difference in x as the width. We will extract from 
the original image the pixels that lie within the boundary lines and place them in the array of zeros.
*Note: Since this crop needs to include full color we need to create our zeros array with shape of (height,width,3)
What is the best way to overlay the cropped section onto the zeros array?
    Ideas: 
    - BF iteratation
    - Numpy slicing (calculate the rounded indicies of each boundary line and then take the slice between those two indicies) This will need to be done for each row 
"""
class crop_imgs(object):
    def __init__(self, img=None):
        self.img = img
        self.params = ap.param_manager()
        self.line = None
        self.ln1 = None
        self.ln2 = None         

    def crop(self,line, row_width):
        self.line = line
        self.width = row_width
        # convert the line to slope intercept form
        self._calc_boundary_lines(self.line, self.width)

    # 
    def _calc_np_slice(self,line):
        assert len(line) == 4, "The line must be in the form of [x1,y1,x2,y2]"
        m,b = self._slope_ind_form(line)
        # Creates an array of all the possible x_values within the image eg. [0....1079]
        y_vals = np.arange(self.img.shape[0])
        assert m != 0, "slope cannot be zero"
        x_vals = (y_vals- b)/m
        # find the points that are on the line
        x_vals = np.round(x_vals).astype(int)
        assert x_vals.shape[0] == y_vals.shape[0], "The x_vals and y_vals must be the same shape"
        assert np.max(x_vals) <= self.img.shape[1], "The x_vals must be less than the width of the image"

        slice_arr = self.img[y_vals,x_vals]
        return slice_arr
    
    # This extracts the pixel values form the image that lie within the two boundary lines. Then it overlays that with the zeros array
    def _slice_vals(self,slice1,slice2):
        assert slice1.shape == slice2.shape, "The slices must be the same shape"
        # ensure that all the slices from slice one are less that slice two
        try:
            assert np.all(slice1 <= slice2), "The slices must be in the form of [x1,y1,x2,y2]"
        except AssertionError:
            assert np.all(slice2 <= slice1), "The slices must be in the form of [x1,y1,x2,y2]"
            temp = slice1
            slice1 = slice2
            slice2 = temp

        # subtract 1 from slice1 to ensure that the slice is inclusive. However if the value is already 0 then leave it as 0
        slice1 = np.where(slice1 > 0, slice1-1, slice1)

        x_range = (np.min(slice1),np.max(slice2))
        diff = x_range[1]-x_range[0]

        assert diff > 0, "The difference between the two slices must be greater than 0"

        crop_shape = (self.img.shape[0],diff,self.img.shape[2])
        crop = self._make_zeros(crop_shape)

        # extract the values from the image. *Note this will be a list since there are unequal numbers of pixels in each row
        # impart the values on the crop array
        sliced_vals = []
        for i in np.arange(slice1.shape[0]):
            start = slice1[i]
            end = slice2[i]
            crop[i,start-x_range[0]:end-x_range[0]] = self.img[i,start:end]
            sliced_vals.append(self.img[i,start:end])

        

    # take the line in the from of [x1,y1,x2,y2] and calculate the two boundary lines
    def _calc_boundary_lines(self,line, width):
        round(width)
        x1,y1,x2,y2 = line
        delt_x = width
        self.ln1 = [x1-delt_x,y1,x2-delt_x,y2]
        self.ln2 = [x1+delt_x,y1,x2+delt_x,y2]

    # Round the intercept to the nearest integer
    def _slope_ind_form(self,line, round_int=True):
        x1,y1,x2,y2 = line
        m = (y2-y1)/(x2-x1)
        b = round(y1-m*x1)
        return m,b


    # This will take a line in the form of [x1,y1,x2,y2] and give the index of the line at the given y_val
    def _calc_index(self,line, y_val):
        x1,y1,x2,y2 = line
        gradient = (y2-y1)/(x2-x1)
        x_val = (y_val-y1)/gradient + x1
        return round(x_val)

    # Remember the the image should be in the form of [height,width,number of channels org_img has]
    def _make_zeros(self,shape):
        return np.zeros(shape)

    """
    test the following domains:
    1. The line is horizontal
    2. The line is vertical
    3. The crop is the correct shape
    4. All the types are identical in the array
    5. The line is diagonal
    """
    def testing(self):
        pass    


class test_diagonal_slicing(object):
    # *Note: Please enter the array shape in the order of (x,y) (col,row) as that is also the order in which the lines are processed. The lines should also follow that order
    def __init__(self, arr, line, arr_shape=(10,20)):
        assert len(arr_shape) == 2, "The array shape must be a 2-tuple, list, or numpy array"

        self.arr = np.arange(200).reshape(arr_shape[::-1])


    def slice_lines(self, ln, floor=True):
        assert type(floor) == bool, "floor must be a boolean value"
        # Make a numpy array of 10x20
        m,b = self._slope_ind_form(ln,floor)

        y_vals = np.arange(self.arr.shape[0])
        # catch horizonal lines, but allow verical ones to pass
        assert m != 0, "slope cannot be zero"

        if m == np.inf:
            x_vals = np.ones_like(y_vals)*b
        else:
            x_vals = (y_vals- b)/m

        # find the points that are on the line
        if floor:
            x_vals = np.floor(x_vals).astype(int)
        else:
            x_vals = np.ceil(x_vals).astype(int)
        # x_vals = np.round(x_vals).astype(int)
        pts = np.array([y_vals,x_vals]).T
        # slice_arr = self.arr[y_vals,x_vals]
        return pts


    def disp_res(self, pts, arr=None):
        # Filter the points by the shape of the array if the x or y will be out of range remove it from the list
        res = np.zeros(self.arr.shape) if type(arr) == type(None) else arr; 
        pts = self._get_frame_values(pts) 
        res[pts[:,0],pts[:,1]] = 2
        return res

    def _get_frame_values(self,pts):
        assert pts.shape[1] == 2, "Error: the values passed must be pts with 2 values (x,y)"
        return np.array(list(filter(lambda x: x[0] < self.arr.shape[0] and x[0] >= 0 and x[1] < self.arr.shape[1] and x[1] >= 0, pts)))


    """ 
    Retrieve all the values that lie between the lines including the lines themselves.
    Return: 
        - List of all the values between the lines in their corresponding rows
    Note: This needs to be a list rather than a numpy array since the size may not remain consistent betwwen the lines if they are not parallel.
    """
    def space_between_lines(self, ln1, ln2):
        # get the lower and upper line
        ln1,ln2 = self._check_line_cross(ln1,ln2)

        # pts values are in (row col) format (y,x)
        pts = [self._get_frame_values(self.slice_lines(ln)) for ln in [ln1,ln2]]

        assert pts[0].shape[0] != 0 and pts[1].shape[0] != 0, "One of the lines does not pass through the frame"

        min_start_ind, max_end_ind = min(pts[0][0,0],pts[1][0,0]), max(pts[0][-1,0],pts[1][-1,0])
        assert min_start_ind <= max_end_ind, "The arrays are not sorted correctly"

        # generate the array to fill the values from a given start to end
        gen_fill_arr = lambda val,start,end:np.array([[i,val] for i in range(start,end)]).astype(int)
        
        # fill the arrays with the values if the start index is not the minimum then add those values to the array with the edge as the x-index and do the same for the max value.
        edges = [0,self.arr.shape[1]-1]
        for i ,pt in enumerate(pts):
            if pt[0,0] != min_start_ind:
                pts[i] = np.append(gen_fill_arr(edges[i],min_start_ind,pt[0,0]),pt,axis=0)
                # alt:
                # pts = np.concatenate(gen_fill_arr(vals[i],min_start_ind,pts[0,0]),pts,axis=0)
            if pt[-1,0] != max_end_ind:
                pts[i] = np.append(pts[i], gen_fill_arr(edges[i],pt[-1,0]+1,max_end_ind+1),axis=0)

        assert pts[0].shape[0] == pts[1].shape[0], "Array 1 length {} does not equal array 2 length {}".format(pts[0].shape[0],pts[1].shape[0])
        
        #Stopping point: In the second test case the lines are being ordered in correctly and thus it is causing the list slicing to fail.
        ret = [[self.arr[pts[0][i,0],pts[0][i,1]:pts[1][i,1]+1],[pts[0][i,1],pts[1][i,1]]] for i in range(pts[0].shape[0])]
        # ret = []
        # for i in range(pts[0].shape[0]):
        #     slices = [pts[0][i,1],pts[1][i,1]+1]
        #     assert slices[0] <= slices[1], "The slices are not ordered correctly"
        #     ret.append(self.arr[pts[0][i,0],slices[0]:slices[1]])
        return ret 

    """
    This will determine which line is greater than the other and also ensure that they will not be crossing within the image. 
    To accomplish this, we must check that the the x_value of the line is less than the other at both the top and bottom of the frame.
    """
    def _check_line_cross(self, ln1, ln2):
        m1,b1 = self._slope_ind_form(ln1,True)
        m2,b2 = self._slope_ind_form(ln2,False)

        try:
            x = (b2-b1)/(m1-m2)
            y = m1*x+b1
            print(self.arr.shape[0],y)
            assert 0 >= y or y >= self.arr.shape[0], "The lines are crossing within the image"
        except ZeroDivisionError:
            print("The lines are parallel")

        """
        Determine first and second by the x_value at the mid y-value of the matrix
        """
        mid_y = self.arr.shape[0]//2
        mid1 = (mid_y-b1)/m1; mid2 = (mid_y-b2)/m2
        if mid1 < mid2:
            return ln1,ln2
        elif mid1 == mid2:
            assert m1 == m2, "The lines cannot intersect within the image unless they are collinear"
        return ln2,ln1
        
        
    #  rounds the intercept to the nearest integer
    def _slope_ind_form(self, line, floor):
        x1,y1,x2,y2 = line
        try:
            m = (y2-y1)/(x2-x1)
        except ZeroDivisionError:
            return np.inf,x1
        if floor:
            b = math.floor(y1-m*x1)
        else:
            b = math.ceil(y1-m*x1)
        return m,b

    """
    Cases: 
        1. The lines are collinear
        2. The lines are parallel, but not collinear
        3. The lines are not parallel and they intersect outside the image
        4. The lines are not parallel and they intersect within the image
        5. The lines converge towards the bottom of the image.
        6. The lines converge towards the top of the image.
        7. The lines intersect at the top of the image.
        8. The lines intersect at the bottom of the image.
    * Note all values corresponding to the space are in (x,y) (col,row) format
    Test is made to function on the default array shape of (10,20)
    """
    def test_values_between(self):
        cases = [
            [[0,0,10,10],[0,0,10,10]],
            [[0,0,10,10],[0,10,10,20]],
            [[0,0,10,20],[5,0,11,20]],
            [[0,0,10,20],[5,0,5,20]], 
            [[0,0,10,20],[5,0,20,20]],
            [[0,0,10,20],[5,0,10,20]],
            [[0,0,10,20],[0,0,11,20]],
            [[3,0,10,20],[7,0,0,20]] # Case 8 should fail
        ]
        for i, case in enumerate(cases):
            print(self.disp_res(self.slice_lines(case[1]), arr = self.disp_res(self.slice_lines(case[0]))))
            # self._check_line_cross(case[0],case[1])
            try: 
                self.space_between_lines(case[0],case[1])
            except AssertionError as e:
                print("Case {} failed on {} error".format(i+1, e.args[0]))

    
    def test_slicing(self):
        test_lines = [
            [0,0,10,20],
            [0,0,10,10],
            [2,0,15,10],
            [5,1,0,10],
            [6,0,1,10]
        ]


        for ln in test_lines:
            # calc the slope-intercept of the line
            pts = self.slice_lines(ln)
            print(self.disp_res(pts),"\n\n", pts, "\n")

class full_testing(object):
    def __doc__(self):
        return "This will provide full testing with masked images of many kinds and multiple display and visualization options."
    
    def __init__(self):
        # **Note all paths here can be relative or absolute
        self.test_paths = [
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\python_soybean_c\\Test_imgs\\Special_tests",
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\corn\\imgs",
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\Drone_images\\Winter Wheat",
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\sm_soybeans\\imgs",
            "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\sm_soybeans\\Quite_weedy"
        ] 
        self.image_types = [".jpg",".png",".jpeg",".tiff",".JPG",".PNG",".JPEG"]
    
    def _load_imgs(self,path, func = None):
        abs_path - os.path.abspath(path)
        files = os.listdir(abs_path)
        test_f_type = lambda x: os.path.splitext(x)[1] in self.image_types
        for file in files:
            if test_f_type(file):
                img = cv2.imread(os.path.join(abs_path,file))
                if func == None: 
                    yield img
                else:
                    yield func(img)




    
if __name__ == "__main__":
    # from Modules.prep_input import interpetInput as prep 
    # from Modules.display_frames import display_frames as disp
    cropper = crop_imgs()
    # load imgs:
    path = "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\corn\\imgs"
    abs_path = os.path.abspath(path)
    for img in os.listdir(abs_path):
        print(img)
        cropper.img = cv2.imread(os.path.join(abs_path,img))


    
