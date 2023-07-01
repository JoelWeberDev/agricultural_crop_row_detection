'''
 Module outline:
  - Take the predfined spaqcing and the camera parameter and calculate where the rows should exist relative to eachother.
  - Use the absolute positioning of the camera to calculate this distance in pixels and frame location.
  - This will take the z value and width of that specific row to 
  - So it seems that the spacing between parallel lines remains constant as the distance from the camera increases regardless of their position along the x distribution.
'''


import cv2
import numpy as np
import path
import sys,os
from icecream import ic

import matplotlib.pyplot as plt

try:
  import Camera_Calibration as cam
except ModuleNotFoundError:
  import Modules.Camera_Calib.Camera_Calibration as cam
  # print("Module not found, using local copy")


sys.path.append(os.path.abspath(os.path.join('.')))

from Modules.prep_input import interpetInput as prep 
from Modules.display_frames import display_dec as disp
from Modules.Image_Processor import procImgs as proc


class generic(object):
    def __init__(self, img_sz = (720,1280,3)):
      # ic(loaded_cam)
      self.img_sz = img_sz
      self.cam_access = cam.loaded_cam

      vp_name = "vanishing_point{}x{}".format(self.img_sz[0],self.img_sz[1])
      self.vanishing_point= self.cam_access.data[self.cam_access.find_key(vp_name)][vp_name]

      self.vp_calculator = cam.vanishing_point_calculator()

      self.conversion_table = {"m":1000,"mm":1,"cm":10,"km":1000000,"in":25.4,"ft":304.8,"yd":914.4,"mi":1609344, "mm":1}


    # ** Make the error margin an adaptive value
    def estimate_slope(self, point, ret_range=False,err_margin = 30, slope = None): # the error margin may rquire some tuning and thus in the future could be an adaptive value
      # Get the slope of the line between the vanishing point and the point of interest

      # Give a range of acceptabe slopes the line occupying that point can be in. This should be a percenage error and not an absolute value since there could be radical differnces in shallow lines that are trivail
      # for steep lines.

      if slope:
        # point at the y value of the vanishing point 
        try:
          proj_point = (self.vanishing_point[1] - (point[1] - slope*point[0]))/slope
        except ZeroDivisionError or RuntimeWarning:
          Warning.warn("Zero division error in slope calculation")
          return False
        if proj_point > self.vanishing_point[0]-err_margin and proj_point < self.vanishing_point[0]+err_margin:

          return True
        return False

      
      if ret_range:
        ret = np.array([self.calc_slope([self.vanishing_point[0]+(i*err_margin), self.vanishing_point[1]], point) for i in range(-1,2,2)])
        ret.sort()
        return ret
      return (self.vanishing_point[1]-point[1])/(self.vanishing_point[0]-point[0])


    def calc_slope(self, point1, point2):
      # Get the slope of the line between the vanishing point and the point of interest
      slope = (point1[1]-point2[1])/(point1[0]-point2[0]) 
      return slope

    # This function is used to take two points of 2D coordinates and estimate their absolute distance
    def estimate_distance_3D(self, point1,point2, err_range = False, error_margin=0.1):
      # Get the real distance between the two points in the sensor frame.
      proj1 = self.vp_calculator.twoD_to_3D(point1)
      proj2 = self.vp_calculator.twoD_to_3D(point2)
      print(proj1,proj2)
      dist = np.linalg.norm(proj1-proj2)
      # this also returns the components of the disance because there may be some nuance between the horizontal and vertical distance.
      if err_range:
        return np.array([[abs(proj1[0]-proj2[0])*(1+(error_margin*i)) for i in range(-1,2,2)], [abs(proj1[2]-proj2[2])*(1+(error_margin*i)) for i in range(-1,2,2)]])
      return np.array([abs(proj1[0]-proj2[0]), abs(proj1[2]-proj2[2])])

    # Ensure that the point is in 3d
    def get_distance_from_point(self, point,dist ,dir_vect = np.array([1,0,0]), units = "m"):
      # Get the distance from the given point to determine the second relative one 
      # use the 3d to 2d function to get the pixel coordinates of the point
      if type(point) == list:
        point = np.array(point)
      # Ensure that the direction vector is a unit vector
      if np.linalg.norm(dir_vect) != 1:
        dir_vect = dir_vect/np.linalg.norm(dir_vect)

      # Get the distance of the magnitude in direction of the unit vector
      pt_rel  = point + (dir_vect*self.to_mm(dist,units))

      print(point, pt_rel)

      return self.vp_calculator.threeD_to_2D(points=[point,pt_rel])
      
    def to_mm(self, dist, units = "m"):
      if not units in self.conversion_table:
          raise ValueError("Units not in conversion table please use one of the following: " + str(self.conversion_table.keys()) + "")
      return dist*self.conversion_table[units]

    def test_slope(self):
      # Test if the slope of the line between the vanishing point and the point of interest is within the acceptable range.
      im_shape = self.cam_info['sensor_resolution']
      disp_im = np.zeros((im_shape[1],im_shape[0],3), np.uint8)

      disp_pts = (np.rint(self.vanishing_point)).astype(int)
      for i in range(100,im_shape[0],222):

          pt_end = int(self.vanishing_point[1]-((val := self.estimate_slope((i,0)))*self.vanishing_point[0]))  

          cv2.line(disp_im,(0,pt_end),(disp_pts),(0,255,0),5)

      # cv2.line(disp_im,(0,c.im_dimensions[1]//2),(disp_pts[0],disp_pts[1]),(255,0,0),5)
      # cv2.line(disp_im,(0,c.im_dimensions[1]),(disp_pts[0],disp_pts[1]),(255,0,0),5)



      # # display in matplotlib
      plt.subplot(111)
      plt.plot(),plt.imshow(disp_im)
      plt.show()

if __name__ == "__main__":
  from Chess_board_calib import project_pts
  g = generic()
  # print(g.get_distance_from_point(np.array([-574,0,207]), 1149.5, units = "mm"))
  # print(g.estimate_distance_3D((0,720),(1280,720)))
  # print(g.estimate_distance_3D((0,0),(1280,1)))

  ic(project_pts(lines=g.vp_calculator.twoD_to_3D(np.array([[0,720],[1280,720],[0,0],[1280,0]]))))
  