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
    s = (tan_inv(rw/fw))*9+s ER| 0.05 < s < 0.5

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

    Curent function: 
        px_min = (mr*ih)/rn + (mr**2*ih*adj)/rn 
    New Function: 
        avw = average width of row (rwt+rwb)/2
        avgr = average green pixels in a row
        px_thresh = (avgr/avw)*1-abs(2s)
    absolute minimum: 

   
"""



def imp_mods(modules):
    import warnings
    pip_reqs = {"ic"}
    local_reqs = {"json"}
    # This is where all the classes or functions of a module will be located. It will point to its parent and will simply need to be named as a global variable 
    sub_mods = {}
    # reqs = pip_reqs.union(local_reqs)
    try:
        for req in pip_reqs:            
            globals()[req] = modules[req]
    except KeyError:
        warnings.warn("The module {} is not present".format(req))
    ic(globals())

import sys, os
from icecream import ic
from decimal import Decimal
import math

sys.path.append(os.path.abspath(os.path.join('.')))
from Modules.json_interaction import load_json


# This is used in interaction with the json file
class param_manager(object):
    def __init__(self):
        self.data = load_json('Adaptive_params/Adaptive_values.json',dict_data=True)

    def access(self, key):
        return self.data.find_key(key, ret_val=True)

    def update(self, key, val,title=None):
        # if type(val) == dict:
        #     for k,v in val.items():
        #         self.update(k,v,key)
        try:
            self.data.write_json({key:val},data_title=title)
        except Exception as e:
            # ic(key, val, title, e)
            self.data.write_json({key:val},data_title=title)

pm = param_manager()


# sensitivity maps from -0.5 to 0.5
class value_calculations(object):
    def __init__(self):
        self.sens = pm.access("sensitivity")
        self.row_count = pm.access("row_count")
        self.rel_bot = pm.access("avg_bot_width_pxls")
        self.rel_top = pm.access("avg_top_width_pxls")
        self.spacing = pm.access("avg_spacing_pxls")
        self.plant_size = pm.access("plant_size")
        self.unit = pm.access("unit")
        # All the green points that were included in the annotations
        self.mes_green = pm.access("annotated_green")
        self.total_green = pm.access("all_green")
        self.annot_pts = pm.access("all_annot_pts")

        self.im_wid,self.im_height = pm.access("img_dims")[0],pm.access("img_dims")[1]

        self.avg_annot_green = self.calc_avgs(self.mes_green)
        self.avg_tot_pts = self.calc_avgs(self.annot_pts)
        self.avg_tot_green = self.calc_avgs(self.total_green)

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

    def slope_perc(self,m=9):
        val = (math.atan(self.rel_bot/self.im_wid))*(m+self.sens)
        return self._verify_range(val,0.05,0.5)
        
    def inlier_coeff(self):
        val = (self.rel_bot/2/self.spacing)*(1+self.sens)
        return self._verify_range(val,0.1,0.5)

    def width_adj(self,wid):
        val = 0.5*math.sqrt(self.spacing)*math.log(wid)*(1+self.sens)
        if self._verify_range(val/self.spacing,0.1,0.75,cond=True) == "High":
            return self.spacing*0.4
        return self._verify_range(val/wid,0.1,0.75)

    def row_pixels(self):
        val = (self.avg_annot_green/((self.rel_bot+self.rel_top)/2))*(1-abs(self.sens*2))
        min_val = math.log(self.avg_annot_green,3) 
        if val < min_val:
            return min_val
        return val
    

    """
    The following parameters need to be determined:
    - 
    """
    def hough_params(self):
        pass
    





if __name__ == "__main__":
    import sys, os 
    from icecream import ic
    js = param_manager()
    ic(js.access("Focal_length"))
    js.update("Focal_length", 1.3, "Camera")