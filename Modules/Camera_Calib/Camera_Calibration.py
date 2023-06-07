''' Camara Calibation
 Description: Calibrate the camara by using a chessboard and the OpenCV library to determine the 
              intrinsic and extrinsic parameters of the camara and the focal length of the camera
 Module outline:
 - Example images: sourced form ==> folder or live capture functionality
 Module plan:
  - Parameters from camera:
   * Focal length
   * Field of view
   * Aspect ratio
   * Sensor resolution (pixels)
   * Sensor resolution (mm)
  - Absolute parameters of camera position:
   * Mount height from neutral point
   * Mount angle from neutral point
  - Potential alternative parameters:
   * The the dimensions of a resized image
   * Gyroscopic data for more precise function on hilly terrain
 - First step: Calculate the z distance from camera to the ground at a given place on the lens
'''

# dependencies:
import cv2
import numpy as np
import json
from icecream import ic 
'''
Completed tasks:
 - load the camera parameters from a json file
 - Setup the initail camera transformation matrix and rotation matrix
 - Build a matrix multiplication function
 - Calculate the projection matrix
 - Calculate the focal length of the camera pixels
'''
import math
import matplotlib.pyplot as plt
import sys,os
import warnings

try: 
    import Chess_board_calib as chess
except ModuleNotFoundError:
    from Modules.Camera_Calib import Chess_board_calib as chess

try: 
    sys.path.append(os.path.abspath(os.path.join('.')))
    from json_interaction import load_json
except ModuleNotFoundError or ImportError:
    sys.path.append(os.path.abspath(os.path.join('.')))
    from Modules.json_interaction import load_json

# loaded_cam = load_json('C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Cascade_Classifier/Modules/Camera_Calib/Camara_specifications.json')
# camera_data = loaded_cam.data[0]
loaded_cam = load_json("C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Cascade_Classifier/Modules/Camera_Calib/Cam_specs_dict.json",dict_data=True)
camera_data = loaded_cam.data["camera_vals"]

class formal_calib(object): 
    def __init__(self):
        self.cam_const = camera_data['focal_length']
        self.mount_height = camera_data['mount_height']
        self.pixels_per_mm = camera_data['sensor_resolution'][0]/camera_data['aspect_ratio'][0]
        self.mount_angle = math.radians(camera_data['mount_angle'])

        self.calc_projection()
        # angles of the camera: X==> pitch Y ==> yaw Z ==> roll
    # What is required for the camera calculations??
    # - The focal length of the camera
    # - The mount angle of the camera
    # - The mount height of the camera
    # - Eventually the gyroscopic data from the camera and the on board sensor.

    def mult_matrix(self,mat_lst = [],n=0):
        if n >= len(mat_lst)-1:
            try:
                return mat_lst[n]   
            except IndexError:
                raise IndexError('The list is empty')
        return np.matmul(mat_lst[n],self.mult_matrix(mat_lst,n+1))
    
    def calc_projection(self):
         
        self.extring = extrinsics()
        self.inring = intrinsics()

        self.p1 = self.mult_matrix([self.inring.iX, self.extring.params[:3,:3]])
        self.p2 = self.mat_vector_mult(self.inring.iX,self.extring.params[:3,3] ,3)
    
    def update_params(self):
        self.calc_projection()
        calib_params = {'intrinsic_params':self.inring.iX,'extrinsic_params':self.extring.params}
        loaded_cam.write_json(calib_params)
    
    def mat_vector_mult(self, mat, vec , iters = None):
        if iters == None: 
            iters = len(vec)
        return np.array([np.dot(mat[i], vec) for i in range(iters)])

class extrinsics(formal_calib):
    def __init__(self):
        # The mount height of the camera in mm
        self.cam_height = camera_data['mount_height']

        # angles of the camera: X==> pitch Y ==> yaw Z ==> roll
        self.pitch= math.radians(camera_data['mount_angle'])
        self.yaw = 0
        self.roll = 0

        self.calc_rotation()
        self.calc_translation()

        # self.params = self.mult_matrix([self.eRot,self.eTrans])
        self.params = np.concatenate((self.eRot,self.eTrans.T),axis=1)

    def calc_rotation(self):
        self.eX = np.array([[1,0,0],[0,math.cos(self.pitch),-math.sin(self.pitch)],[0,math.sin(self.pitch),math.cos(self.pitch)]]) 
        self.ey = np.array([[math.cos(self.yaw),0,math.sin(self.yaw)],[0,1,0],[-math.sin(self.yaw),0,math.cos(self.yaw)]])
        self.eZ = np.array([[math.cos(self.roll),-math.sin(self.roll),0],[math.sin(self.roll),math.cos(self.roll),0],[0,0,1]])
 
        self.eRot = super().mult_matrix([self.eX,self.ey,self.eZ])

    def calc_translation(self):
        self.eTrans = np.array([[0,0,0]])

class intrinsics(formal_calib):
    def __init__(self):

        self.cam_const = camera_data['focal_length']
        self.cam_res = camera_data['sensor_resolution']
        self.cam_aspect = camera_data['aspect_ratio']
        self.cam_fov = camera_data['field_of_view']

        self.pxl_per_mm = self.cam_res[0]/self.cam_aspect[0]

        self.calc_intrinsic()

    def calc_intrinsic(self):
        # self.cam_const_pxl = self.cam_const*self.pxl_per_mm
        # self.iX = np.array([[self.cam_const_pxl,0,self.cam_res[0]/2],[0,self.cam_const_pxl,self.cam_res[1]/2],[0,0,1]]) 

        self.iX, self.dist,img = chess.calibrate()
        # MM
        # self.iX = np.array([[self.cam_const_pxl,0,self.cam_res[0]/(2*self.pxl_per_mm)],[0,self.cam_const_pxl,self.cam_res[1]/(2*self.pxl_per_mm)],[0,0,1]])
        



# This class calcualtes the vanishing point of a given image
# This is done by assuming that the intersection plane is flat and that the camera is mounted at a fixed height
# Two parallel lines are imagined and two points are taken from each line and projected onto the image plane
# The vanishing point is then calculated by taking the intersection of the two lines that are formed by the two points
class vanishing_point_calculator(formal_calib):
    def __init__(self):
        '''
        Test possibilities: Use differing points to estimate the error between the vanishing point calculated and the actual vanishing point
        these test values if completely accurate should produce a consistent vanishing point every time
        '''
        self.img_sz = (720,1280,3)
        self.parameters = formal_calib()
        self.lines = np.array([[100,0,1],[100,0,10],[-10,0,10],[-10,0,100]]) 

        self.extrinsic = np.array(loaded_cam.find_key('extrinsic_params',True))
        # ic(self.extrinsic)
        # self.intrinsic = np.array(loaded_cam.find_key('intrinsic_params',True))
        self.intrinsic = self.parameters.inring.iX

        self.rot = self.extrinsic[:3,:3]
        self.trans = self.extrinsic[:3,3]

        ''' Update the vanishing point '''
        # self.update_vp()

    def update_vp(self):
        self.parameters.calc_projection()
        self.calc_vp()

        ''' Writing the vanishing point to the JSON file '''        
        loaded_cam.write_json({'vanishing_point{}x{}'.format(self.img_sz[0],self.img_sz[1]):self.vanish_point},data_title='calc_values')


    def threeD_to_2D(self,points = None):
        if points == None:
            points = self.lines

        # if points.shape[1] == 3:
            # points = np.append(points,np.ones((points.shape[0],1)),axis=1)

        p1 = self.parameters.p1
        # p2 = self.parameters.p2

        # p2 = np.array([0,1000,0]).T; p2.shape = (3,1)
        p2 = np.array([0, 1000, 0]).T; p2.shape = (3,1)
        coeff = self.parameters.pixels_per_mm

        full_params = np.concatenate((p1,p2),axis=1)

        def solve(point):
            pn = point+p2
            vals = ((self.rot @ pn.T)@self.intrinsic) 
            pxl_pts = np.array([[(vals[0]/vals[2])*coeff,(vals[1]/vals[2])*coeff]])
            return pxl_pts[0]
        
        # return np.array([solve(i) for i in points])

        self.sensor_points = np.array([[(np.dot(ln, p1[i][:3]) +p2[i])/self.parameters.pixels_per_mm  for i in range(3)]for ln in points])
        # self.sensor_points = np.array([[ln @ p1[i][:3] +p2[i]  for i in range(3)]for ln in points])
        pixel_points = np.array([[self.sensor_points[i][0]/self.sensor_points[i][2],self.sensor_points[i][1]/self.sensor_points[i][2]] for i in range(len(self.sensor_points))])

        return pixel_points.ravel()

    def calc_vp(self):
        # calculate the two lines and then the intersection
        # self.pixel_points = self.threeD_to_2D()
        self.pixel_points,self.img_sz = chess.manual_proj(self.rot,self.trans,intr=self.intrinsic,ret_shape=True)
        ic(self.img_sz)
        # self.pixel_points = chess.project_pts()

        ic(self.pixel_points)
        vals_per_line = int(len(self.pixel_points)/2)
        lines = [self.cartesian_lns(self.pixel_points[i:i+(vals_per_line)]) for i in range(0,len(self.pixel_points),vals_per_line)]

        lines = np.reshape(lines,(2,2))

        self.vanish_point = self.line_intersection(lines[0],lines[1])


    def best_fit_line(self, pts):
        """ Returns the best fit line for the given points. """
        x = pts[:, 0]
        y = pts[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]
        ic(m,c)
        return m, c
    
    def mid_val(self, line1,line2):
        img_mid = load_json.find_key("camera_resolution",True)[0]/2
        calc_y = lambda ln: ln[0]*img_mid+ln[1]

        pts = np.array([calc_y(ln) for ln in [line1,line2]]) 
        return np.mean(pts)

    def cartesian_lns(self, line):
        if len(line) > 2:
            return self.best_fit_line(line)

        slope = (line[0][1]-line[1][1])/(line[0][0]-line[1][0])
        intercept = np.mean([line[i][0] - slope*line[i][0] for i in range(len(line))])

        return [slope,intercept]

    def line_intersection(self,line1,line2):
        x_val = (line2[1]-line1[1])/(line1[0]-line2[0])
        return [x_val,(line1[0]*x_val+line1[1])]

    # This is designed to return the point where the ray is at y = 0
    def twoD_to_3D(self,points):

        
        if type(points) == list:
            points = np.array(points)

        if points.shape == (2,):
            points = np.array([points])

        iX = self.intrinsic[0]
        iY = self.intrinsic[1]
        fy,cy = iY[1],iY[2]
        fx,cx = iX[0],iX[2]

            


        # euclidian solution
        def eu_solution(point,Yw = 1000):
            ang = self.parameters.mount_angle 
            Xm = (point[0]-cx)/fx
            Ym = ((point[1]-cy)/fy)+math.sin(ang)
            Ym = (point[1]-cy)/fy
            Zm = 1+(math.cos(ang)-math.sin(ang))
            # Zm = 1

            # ic(Xm,Ym,Zm)

            Xw = (Xm*Yw)/Ym
            Zw = (Zm*Yw)/Ym
            return np.array([Xw,Yw,Zw]).T


        ''' This equation was a theory of how to preform all the calculations in Euclidean space, but it was not correct.'''
        def solution(pt,Yw = 1000): 
            rot_inv = np.linalg.inv(self.rot)
            eX = self.rot[0]
            eY = self.rot[1]
            r22,r23 = eY[1],eY[2]
            r32,r33 = eX[1],eX[2]
            # Yc = (Yw*r22 + Yw*r23)
            # Zc = (Yc*fy)/(pt[1]-cy)
            def get_Zw():
                a = pt[1]
                b = r23
                c = r22*Yw
                d = r33
                e = r32*Yw
                Zw = (a*e+a*cy-fy*c)/(fy*b-a*d)
                return Zw
            Zw = get_Zw()
            Xw = (pt[0]*Zw + Zw*cx)/fx
            # Zc = (Yw*fy)/(r22*pt[1] + fy*r32 + cy*r22)
            # Yc = (Yw - r32*Zc)/r22
            # Xc = (pt[0]*Zc + Zc*cx)/fx
            # pw = np.matmul(np.array([Xc,Yc,Zc]).T , rot_inv)
            pw = np.array([Xw,Yw,Zw]).T
            ic(pw)

        def conventional(point):
            point[1] *= -1
            point[1] += cy*2

            p = point.T; p.shape = (3,1)

            pc = np.linalg.inv(self.intrinsic) @ p

            t_inv = self.trans.T; t_inv.shape = (3,1)

            pw = t_inv + (self.rot @ pc) 

            cam = np.array([0,0,0]).T; cam.shape = (3,1)
            cam_w = t_inv + (self.rot @ cam)

            vect = pw - cam_w
            unit_vect = vect/np.linalg.norm(vect)

            lamb = (self.parameters.mount_height)/unit_vect[1]*-1
            
            return np.array([lamb*unit_vect[0],lamb*unit_vect[1]*-1,lamb*unit_vect[2]]).ravel()


        hom_pts = np.array([np.insert(point,2,1) for point in points])
        return np.array([conventional(point) for point in hom_pts])
        # return conventional()


class camera_cals(object):
    def __init__(self,json_path,im_dimensions=None,radians=False):
        self.load_cam_params(json_path)

        self.ang_unit = radians

        # The field of view is almost always the larger angle of the two
        if self.cam_params['aspect_ratio'][0] > self.cam_params['aspect_ratio'][1]:
            self.xfov = self.calc_fov(self.cam_params['field_of_view'])
            if self.ang_unit:
                self.yfov = self.cam_params['field_of_view']
                self.mount_angle = self.cam_params['mount_angle']
            else:
                self.yfov = self.radians(self.cam_params['field_of_view'])
                self.mount_angle = self.radians(self.cam_params['mount_angle'])


        if im_dimensions is not None:
            self.im_dimensions = im_dimensions
        else:
            self.im_dimensions = (self.cam_params['sensor_resolution'][0],self.cam_params['sensor_resolution'][1])

        self.get_vanishing_point()

    
    # The json file should contain the following parameters:
    # - Focal length
    # - Field of view
    # - Aspect ratio
    # - Sensor resolution (pixels)
    # - Sensor resolution (mm)
    # - Mount height from neutral point (mm)
    # - Mount angle from neutral point degrees defualt
    def load_cam_params(self,json_path):
        with open(json_path) as f:
            self.cam_params = json.load(f)
            f.close()

    # Most cameras only provide a single field of view value. This function will calculate the other field of view value based on the aspect ratio of the camera. The fov is almost always the larger angle of the two.
    def calc_fov(self,fov, axis='y'):
        fov = fov if self.ang_unit else self.radians(fov)
        if axis != 'y':
            return 2*math.atan(math.tan(fov/2)*self.cam_params['aspect_ratio'][0]/self.cam_params['aspect_ratio'][1])
        return 2*math.atan(math.tan(fov/2)*self.cam_params['aspect_ratio'][1]/self.cam_params['aspect_ratio'][0])
    
    def radians(self,degrees):
        return degrees * math.pi / 180

    # The z value is the distance from the camera to the ground at a given place on the image frame indacted by pixels in the y direction  
    # ** Remember that the y axis is inverted in the image frame **
    def get_z(self,y_coor): 
        # pixel_cent = self.im_dimensions[1]/2
        # Should go from 0->1 as the y_coor goes from 0->image height
        pxl_ratio = y_coor/self.im_dimensions[1]

        inc_angle = self.yfov*pxl_ratio+(self.mount_angle-self.yfov/2)

        # The math library uses radians by default so we need to convert the angle to radians if not already in radians

        return (1/math.cos(inc_angle))
    
    # Z is the distance from the camera to the ground at a given place on the image frame indacted by pixels in the y direction
    def get_width(self,z):
        return(math.tan(self.xfov)*z)
    
    def get_vanishing_point(self):
        z1 = self.get_z(0)
        z2 = self.get_z(self.im_dimensions[1]/2)
        w1,w2 = self.get_width(z1),self.get_width(z2)

        # This is the absolute ratio between the two edges.
        delt_w = 1-(w1/w2)
        # Breaking point: My next step is to convert the width to a pixel value and then use that to create a line and finally find the vanishing point on that line.
        org_point = np.array([0,self.im_dimensions[1]])
        point_dest = np.array([(self.im_dimensions[0]*delt_w),0])

        # In the fields the system should always be perpendicular to the crop rows so the vanishing point should be in the middle of the image frame 
        #  **** Exceptions and cases: ****
        #   When the system is on an incline that tilts it in any direction the vanishing point will be off center or closer or further away from the center. This can be accounted for be the intgration of a gyroscope or accelerometer.
        #   When the rows are curverd or planted slightly off pefect this will cause the thoeretical calcualtions to be off as well.
        self.vanishing_point = np.array([self.im_dimensions[0]/2,self.im_dimensions[1]+((point_dest[1]-org_point[1])/point_dest[0])*(self.im_dimensions[0]/2)])

    def get_relative_slope(self, point):
        # Get the slope of the line between the vanishing point and the point of interest
        slope = (self.vanishing_point[1]-point[1])/(self.vanishing_point[0]-point[0]) 

        # Give a range of acceptabe slopes the line occupying that point can be in. This should be a percenage error and not an absolute value since there could be radical differnces in shallow lines that are trivail
        # for steep lines.
        err_margin = 0.1
        return slope
        return np.array([slope-(slope*err_margin),slope+(slope*err_margin)]) 

    def calc_spacing(self, p1,p2):
        # Get the the slope of each line from the points to the vanishing point
        sl1,sl2  = self.get_relative_slope(p1),self.get_relative_slope(p2)


# Load images:

# apply the camara calibration function to the images 
def calib(imgList):
    boardSize = (8,8)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((boardSize[0]*boardSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:boardSize[0],0:boardSize[1]].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.   
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for im in imgList:
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, boardSize,None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            im = cv2.drawChessboardCorners(im, boardSize, corners2,ret)

def conventional_calculator():

    c=camera_cals('C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Cascade_Classifier/Modules/Camera_Calib/Camara_specifications.json')
    disp_im = np.zeros((c.im_dimensions[1],c.im_dimensions[0],3), np.uint8)

    disp_pts = (np.rint(c.vanishing_point)).astype(int)
    prev_int = 0
    for i in range(100,c.im_dimensions[0],222):
        pt_end = int(c.vanishing_point[1]-(c.get_relative_slope((i,0))*c.vanishing_point[0]))  
        inter=(pt_end*-1)/c.get_relative_slope((i,0))
        prev_int = inter
        cv2.line(disp_im,(0,pt_end),(disp_pts),(0,255,0),5)
    # cv2.line(disp_im,(0,c.im_dimensions[1]//2),(disp_pts[0],disp_pts[1]),(255,0,0),5)
    # cv2.line(disp_im,(0,c.im_dimensions[1]),(disp_pts[0],disp_pts[1]),(255,0,0),5)

     # # display in matplotlib
    plt.subplot(111)
    plt.plot(),plt.imshow(disp_im)
    plt.show()

def formal_calculator():
    vp_calc = vanishing_point_calculator()
    vp_calc.update_vp()
    print(vp_calc.vanish_point)



if __name__ == "__main__":
    # conventional_calculator()
    formal_calculator()
