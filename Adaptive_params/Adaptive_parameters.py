"""
Author: Joel Weber
Title Apdaptive_parameters
Overview: This module manages and updates the constants that could have varing values based on the conditions
Vision: This will be the hub of the machine learning portion where the optimal values will be found added to the automatic detection model
"""

"""
Module outline:
 Purpose: Through this we should be able to make all the values coincide with eachother the fashon best suited for the conditions of hte image
 
 Tasks:
  - Tie any quantity that may directly vary with the environmental conditions to this module
  - With the tie we must create a fucntions that will change the value based on user inputs and a sesitivity variable.
   - How will these functions act and be constructed? 
    - Is it trial and error or for the best results and how do we avoid simply making more magic numbers that act as coefficients?
    - How  much metadata are we required to take from the user? 
     - We could add metadata as a means to calibrate the system more accurately
     - This could be used with the option of a row annotation where the user will draw a bounding box around the rows for this will allow us to calculate the row's spacing width, color, average height, number of rows
      along with other values.

 Helpful values to calcualte:
  - Obtain a relative noise or weed density value
   - This could be done by taking the number of all green pixels and the calculated theoretical 
    according to the user defined values

 Example: Iniler coefficient
  This is the value that governs the precision at which the lines are placed relative to the 
  pixels between the rows and groups the lines accordingly. It is a percentage of the displacement
  distance that has been calculated in pixels. 
  What varialbles will neccessitate a change in this value?
   - Row spacing (it can be looser as the dispalcment value gets looser)
   - Row width (Since it should cover any point on the row this values should be at least half the 
    width of the row)
   

 Modules:
  Hough_values:
  User_annotation:
   - This is where the user is shown the frame and they are asked to calibrate the system by
   drawing a bounding box around the rows. With this we can verify and error correct with the 
   theoretical values that have been calculated.

Provided values and dependent varaiables:
 Provied / absolutely calculated values:
 - Row spacing (mm)
 - Row spacing (pixels)
 - Row width (mm)
 - Row width (pixels)
 - Row center (top and bottom pixels)
 - Row center (top and bottom mm)
 - Row Number **nuance this may not be fully represenative for there may be partial rows on the perhiveral of the image that are not annotated
 - Slope 
 - Number of green pixels in the row
 - Number of green pixels in the image
 - Number of green pixels in the rows combined (total) **nuance this also may be slightly skewed for there could be partial rows that were not annotated by the user
 - Image size (pixels)
 - Kernel that was prefromed on the image
 - Camera parameters 

 Intermediary variables:
 - Concentration of green pixels in the row
 - Approxomate of the green noise within the image (pixels)
 - Theoretical number of green pixels in the row
 - 
 
 Dependent variables:
 - Inlier coefficient (bottom)
 - Slope error margin
 - Row width error margin
 - Row spacing error margin
 - Number of points that define a line within the row 
 - Estimated row spacing
 - Estimated row width
 - The max line gap based on the green pixels concentration within the image
 - Number of incremenatal masks
 - Error margin for the horizonatal variance of a good line from the vanishing point

If there is a very poor amount of green that lies within the row the implication it that the annotater did an aweful job or we have a high weed concentration. In this case we must make the system more robust such 
that we will not be tricked by this. The following measures will be taken:
 1. Make the incremental masks tighter and more nurmeous
 2. Make the inlier coefficient smaller ** This may make splits to be made where it is inapropriate
 3. Make the slope error margin smaller
 4. Narrow the max spacing gap

Functions for variables:
 Inlier coefficient:
 0.1 - 0.5
  - This is the fraction that determines what percentage of the disparity between two lines. This value should be capped at 0.5 for if it exceeds that there will be overlap in the range that lines are grouped
  This value is related to the following independant variables:
    rs Row spacing (directly proportional)
    rw Row width (pixels) (directly proportional)(Most critical)
    s overall sensitivity of the module
    e the error margin of slope
    Get the ratio beteen the spacing and width pixels and give that the 
    c = (rw/2/rs)*1+s ER| 0.1 < c < 0.5 

 Slope error margin:
  Where is this value used? : Group_line ln 90
  This basically is the measure of how far the line can be from the vanishing point. 
  An ideal slope is identified and it is contrasted with the real slope of the line. 
  It is the primary filter for the good and bad lines.
  There may be little need for variance here
  Independent variables:
    rw Row width (pixels) (directly proportional)
    s Model sensitivity (directly proportional) correlation: 0.10
    fw frame width (pixels) (inversely proportional)
    s = (tan_inv(rw/fw))*9+s ER| 0.05 < s < 0.65

 Row width error margin:
  This is included because the rows in the image that was annotated may not be fully representative
  of the real row width.
  Use cases: Adaptive_parameters 170
    Independent variables:
        rw Row width (pixels) (directly proportional)
        s Model sensitivity (directly proportional) correlation: 0.10
        rs Row spacing (pixels) (directly proportional)
        rwa = 0.5sqrt(rs)*log(rw)*1+s ER| 0.1 < rwa < 0.75

 Row spacing error margin:
  There may be nuances in how the spacing has been estimated
  Use cases: Adaptive_parameters 170
    Independent variables:
        rs Row spacing (pixels) (directly proportional)
        s Model sensitivity (directly proportional) correlation: 0.10
        rwc Row error margin (directly proportional)
        rw Row width (pixels) (directly proportional)
        rsc = rs-(rwc*rw)*1+s ER| 0.1 < rsc < 0.40

 Number of points that define a line within the row:
  This is for the hough lines and it will be used to determine the number of points that lie 
  on an average single line in the row. This one is fairly crucial to be accurate with for there 
  exisits problems both tight the upper and lower bounds. If it is too tight then the lines will
  not form correcly on sparase rows and if too loose then lines will form on the noise.

  How should the number of points per line scale with the total amount of green pixels in the row
    
  Use cases: incremental_masks ln:57
    Independent variables:
        rs Row spacing (pixels) (directly proportional)
        s Model sensitivity (directly proportional) correlation: 0.10
        rw Row width (pixels) (directly proportional)
        gr green pixels in the row compared to the total number of pixels in the row (directly proportional)
        mr Pixel ratio in the curent mask to the overall mask
        ih image height (pixels) 
        rn Row number
        adj adjustment factor (What we need to determine in adaptive_parameters)
        gr_err the ratio of annotated green to total green 
        agp average green pixels in the annotated rows

    Function:
        rp = 1.5log1.3((agp*gr_err)/rw)*1+s
    absolute minimum: 

 Max line gap:
  This is the third parameter of the hough line transform. It will prevent clustering from defining
  a line. This value will only be fration that then can be applied to the image height.
  Use cases: incremental_masks ln:57
  Err towards the higher end rather than the lower
  Verify that it is lower than the min line len

  Independant variables: 
    gr ratio of green pixels within the annotated row to the total that were encapsulated by the 
     annotation (inversely proportional) 0 < gr < 1
    rw row width (inversely proportional)
  Function: 
    mxg = 1/(gr+log20((iw/640)rw)) * 1+(s/2)

 Min line length:
    This is the fourth parameter of the hough line transform. It will prevent clustering from defining
    a line. This value will only be fration that then can be applied to the image height.
    Use cases: incremental_masks ln:57

    Independant variables:
        gr ratio of green pixels within the annotated row to the total that were encapsulated by the

    Const: The fraction of the row that is the minimum line length
    Function:

 Incremental mask:
    The amount of masks that are created in the image semenation process this is basically the 
    resolution at which the image is processed.

    independant variables:
        tg total number of green pixels in the image

    icm = int(log3.5(tg) + s) 5 < icm < 11

 Vanishing point error margin:
    This is the error margin that is used to determine if a line is close enough to the vanishing point
    It will be multiplied by the image width to determine the allowed variance in pixels.

    independant variables:

 Color range for incremental masks:
    This is the range of green hue that each of the masks will be split into and thus we need to determine how liberal to make this
    higher when there are fewer points
    tgr total green in the image (inversely proportional)

    cr = int(25log15(tgr)*1-(s/2)) 6 < cr < 10

 Weight of missing line:
    Description: I a video or a linearly coherent set of images we can improve the accuracy and consistency of the model by processing the previous frame to determine which groups of lines best match the new ones
    However if a line is missed in the frame that should decrease the probability of it being the correct set of lines thus we need to assign a wieght to the missing line.
    Use cases: consider_prev_lines

    independant variables:
     - rn detected row number
     - spacing between rows
     - max number of rows

Kernel size:
    Description: Determine from spacing and row width both the vertical and horizontal kernel size 

    Independant variables:
        - rw row width
        - s model sensitivity
        - cr Coefficient of reduction: This is a hard-coded value that will simply take reduce the size of the kernel by a specific ratio

    Function:
        - ksv = int(rw*cr)*1+s


"""



# def imp_mods(modules):
#     import warnings
#     pip_reqs = {"ic"}
#     local_reqs = {"json"}
#     # This is where all the classes or functions of a module will be located. It will point to its parent and will simply need to be named as a global variable 
#     sub_mods = {}
#     # reqs = pip_reqs.union(local_reqs)
#     try:
#         for req in pip_reqs:            
#             globals()[req] = modules[req]
#     except KeyError:
#         warnings.warn("The module {} is not present".format(req))
#     ic(globals())

import sys, os
from icecream import ic
from decimal import Decimal
import math

sys.path.append(os.path.abspath(os.path.join('.')))
from Modules.json_interaction import load_json


# This is used in interaction with the json file
# This path is globle since there are many differnet path files and tests may need to use other paramters than the default 
path = 'Adaptive_params/Adaptive_values.json'
# The uiversal settings simply assist in ensuring that the correct parameters are loaded when there are other preconfigured json files with a testing data set
class param_manager(object):
    def __init__(self):
        uni_path = "Adaptive_params/parent_setting.json"
        assert os.path.exists(uni_path), "The path {uni_path} does not exist".format(path)

        self.universal_settings = load_json(uni_path,dict_data=True)
        self.path = self.universal_settings.find_key("parameter path", ret_val=True)
        try:
            assert os.path.exists(self.path), "The path {} does not exist".format(self.path)
        except AssertionError:
            self.universal_settings.write_json({"parameter path": path}, "Current Settings")
            self.path = self.access("parameter path", universal=True)

        self.data = load_json(self.path,dict_data=True)

    def access(self, key, universal=False):
        data = self.universal_settings if universal else self.new_data()
        val = data.find_key(key, ret_val=True)

        
        assert val != -1, "The key '{}' does not exist".format(key)
        return val

    def update(self, key, val,title=None, universal=False):
        data = self.universal_settings if universal else self.new_data()
        try:
            data.write_json({key:val},data_title=title)
        except Exception as e:
            data.write_json({key:val},data_title=title)

        if universal and key == "parameter path":
            self.new_data() 
            assert os.path.exists(val), "The path {} does not exist".format(val)
            self.path = val

    def new_data(self):
        latest_path = self.universal_settings.find_key("parameter path", ret_val=True)
        if self.path != latest_path:
            ic(latest_path, self.path)
            self.path = latest_path
            self.data = load_json(self.path,dict_data=True)
        return self.data

from Modules import json_interaction as ji
pm = param_manager()



# sensitivity maps from -0.5 to 0.5
class value_calculations(object):
    def __init__(self,ui=False,efficiency=False):
        self.im_wid,self.im_height = pm.access("im_dims")[0],pm.access("im_dims")[1]
        # assert self.im_wid == img.shape[1] and self.im_height == img.shape[0], "The image dimensions do not match the ones in the json file"
        if ui:
            self._verify_spacings()

        self.load_hough_vals()
        if not efficiency:
            self.load_vals()


    def load_hough_vals(self):
        self.err_green = pm.access("err_green")
        self.sense = pm.access("sensitivity")
        self.row_count = pm.access("row_num")
        self.rel_bot = pm.access("avg_bot_width_pxl")
        self.rel_top = pm.access("avg_top_width_pxl")
        self.avg_row_wid = (self.rel_bot+self.rel_top)/2

        self.mes_green = pm.access("annotated_green")
        self.annot_pts = pm.access("all_annot_pts")
        self.green_ratio = self.calc_avgs(self.mes_green,self.annot_pts)

    def determine_hough(self,tot_pts):
        return (self.row_pixels(tot_pts),self.min_line_length(),self.max_line_gap())

    def load_vals(self):

        self.calc_vals = {
            "slope_error":self.slope_perc,
            "inlier_coeff":self.inlier_coeff,
            # "width_adjust":self.width_adj,
            # "row_pixels":self.row_pixels,
            "max_line_gap":self.max_line_gap,
            "min_line_len":self.min_line_length,
            "incremental_masks":self.incremental_masks,
            "vanishing_point_error":self.vanishing_point_error,
            "err_green":self.annot_green_err,
            "color_range":self.inc_color_steps
        }

        self.total_green = pm.access("all_green")
        
        self.avg_annot_green = self.calc_avgs(self.mes_green)
        self.avg_tot_pts = self.calc_avgs(self.annot_pts)
        self.avg_tot_green = self.calc_avgs(self.total_green)
        self.noise_ratio = self.calc_avgs(self.mes_green,self.total_green)

        self.spacing = pm.access("avg_spacing_pxls")
        self.plant_size = pm.access("plant_size")
        self.unit = pm.access("unit")
        # All the green points that were included in the annotations

        self.adj_bot_coeff = self.width_adj(self.rel_bot)


        
    # Takes the average of the value given and divides it by the denotmontaor liste or the number of rows if no denomonator is provided
    def calc_avgs(self,val,denom=None):
        if denom == None:
            denom = self.row_count
        return val/denom
    
    def _verify_range(self,val,low,high,cond=False):
        if val > high:
            if cond:
                return "High" 
            return high
        elif val < low:
            if cond:
                return "Low"
            return low
        else:
            if cond:
                return "Good" 
            return val

    def _verify_spacings(self):
        theor_spacing = pm.access("spacing")
        abs_spacing = pm.access("avg_spacing_mm")
        return self.perc_diff(abs_spacing,theor_spacing)

    def write_json(self):
        for key,val in self.calc_vals.items():
            pm.update(key,val(),title="Adaptive_values")

    def slope_perc(self,m=9):
        val = (math.atan(self.rel_bot/self.im_wid))*(m+self.sense)
        return self._verify_range(val,0.05,0.65)
        
    def inlier_coeff(self):
        # val = (self.rel_bot/2/self.spacing)*(1+self.sense)
        val = (self.rel_bot/1.5/self.spacing)*(1+self.sense)
        return self._verify_range(val,0.1,0.5)

    def width_adj(self,wid):
        val = 0.5*math.sqrt(self.spacing)*math.log(wid)*(1+self.sense)
        if self._verify_range(val/self.spacing,0.1,0.75,cond=True) == "High":
            return self.spacing*0.4
        return self._verify_range(val/wid,0.1,0.75)

    # The calculations for this value is actually done in incremental_masks.py since there are factors 
    # in each individual mask that should be processed and taken into account
    def row_pixels(self,gr_pts):
        gr_pts_rows = gr_pts/self.row_count
        ic(gr_pts_rows,self.err_green,self.avg_row_wid,self.sense)
        val = round(math.log((8*(gr_pts_rows*20*self.err_green)/self.avg_row_wid)+(self.avg_row_wid/1000),1.5)*(1-(self.sense/2))) if gr_pts> 300 else 3

        if self.avg_row_wid < 10:
            val /= 2
        # ic(self.avg_row_wid,self.avg_annot_green,self.sense,self.row_count)
        # val = (self.avg_annot_green/((self.avg_row_wid)/2))*(1-abs(self.sense*2))
        return self._verify_range(val, 5,self.im_height/2)

    # This calculation should be done in incremental_masks.py since the value is dependent on the number of masks
    def max_line_gap(self):
        val = self._verify_range(self.min_line_length()/2,0.2,0.4)
        return round(val*self.im_height)

    def min_line_length(self):    
        val = self._verify_range(1/(self.green_ratio + math.log((self.im_wid/640)*self.avg_row_wid,15))*(1+(self.sense/2)),0.3,0.8)
        # val = self._verify_range(self.green_ratio, 0.3,0.8)*(1+(self.sense/2))
        return round(val*self.im_height)

    # Change the log value to make the number of masks more or less
    def incremental_masks(self):
        return self._verify_range(round(math.log(self.total_green,1.8)+self.sense),7,21)

    def inc_color_steps(self):
        val = round(math.log(30,self.total_green)*25*(1+(self.sense/2)))
        ic(val)
        return self._verify_range(val,6,12)

    def vanishing_point_error(self):
        val = (self.rel_bot*1+self.sense)*1.15 # ** Perhaps some action may be neccessary here on the 
        # integration of the sensor values to ensure that a negative numer will not be excessivly 
        # rigourous
        return val

    def annot_green_err(self):
        val = self.mes_green/self.total_green
        return self._verify_range(val,0.3,1) 

    def perc_diff(self, val1, val2):
        return abs(val1-val2)/val1


def annotations_for_test(test_path):
    from Modules.image_annotation import main 
    

class testing(object):
    def __init__(self):
        from Modules.save_results import data_save as save
        self.saver = save()
        self.test_paths = [
            "Adaptive_params\\tests\\small_corn",
            "Adaptive_params\\tests\\mid_corn",
            "Adaptive_params\\tests\\small_soybeans"
        ]    
        self.params = param_manager()

    def image_tests(self):
        from Modules.image_annotation import main 
        from Algorithims_tst.prim_detector import inc_color_mask
        import shutil 
        import random
        for test in self.test_paths:
            imgs = prep("sample", os.path.join(test,"imgs"))
            json_file = self.contains_json(test)

            if json_file != None:
                js_path = os.path.join(test, json_file)
                # pm = param_manager(json_file)
            else:
                # ic(imgs)
                json_path = main(random.choice(imgs["imgs"]))
                new_path = os.path.join(test , "test.json")
                shutil.copy(json_path, new_path)
                assert os.path.exists(new_path), "Json file was not copied correctly"
                js_path = new_path

            self.params.update("parameter path", js_path, title="Current Settings", universal=True)
            self._test_json_loads(js_path)
            samples = disp(imgs, inc_color_mask)
            # samples = disp(imgs, inc_color_mask, disp_cmd = "final")
            # ic(samples)
        global path
        # self.params.update("alternative_path", path, title="user_input") 

    def contains_json(self,path):
        for file in os.listdir(path):
            if file.endswith(".json"):
                return file 
        return None 

    def save_results(self,results, path, s_type = "imgs" ):
        self.saver.save_data(results, s_type, path, "row_detection_results" )

    def _test_json_loads(self, cur_json):
        param_path = os.path.abspath(os.path.realpath(self.params.path))
        desired_path = os.path.abspath(os.path.realpath(cur_json))
        alleged_path = os.path.abspath(os.path.realpath(self.params.access("parameter path",universal=True)))
        ic(param_path, desired_path, alleged_path)
        assert param_path == desired_path == alleged_path, "Current path json file and parameter path do not match" 
        


        
def test():
    test = testing()        
    test.image_tests()

# Generate some tests that will determine if there are issues in accessing the correct json file between modules.  
"""
    1. Test that the json file is loaded correctly from the image annotation module
    2. Switch the parameters from the parent json.
    3. Determine if the parameters that are being processed are the same as the ones in the parent json file.
    4. Check correct path update on accessing a parameter.
"""
def test_parameter_switch():
    pass
    


if __name__ == "__main__":
    import system_operations as sys_op
    sys_op.system_reset()
    import sys, os 
    from icecream import ic
    from Modules.prep_input import interpetInput as prep 
    from Modules.display_frames import display_dec as disp
    from Modules.save_results import data_save as save

    # js = param_manager()
    # vc = value_calculations()    
    # vc.write_json()
    test()
    import system_operations as sys_ops
    sys_ops.system_reset()



