import numpy as np
import cv2
import glob
import math

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(dirpath='C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Images/Chess_imgs/*.jpg', prefix='calib', image_format='jpg', square_size=32, width=9, height=6):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(dirpath)

    shape =0

    for fname in images:
        img = cv2.imread(fname)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

        ''' Display the images with the corners drawn on them.'''    
        # cv2.imshow("title",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

    return mtx, dist, img.shape

def project_pts(lines = np.array([[10,1000,1],[10,1000,10],[-10,1000,1],[-10,1000,10]],dtype=np.float32),mat = calibrate()[0],dist = calibrate()[1],ret_shape = False):
    # mat, dist = calibrate('Modules/Camera_Calib/Chess_imgs/*.jpg', 'calib', 'jpg')
    rvec = np.array([math.radians(60), 0,0],dtype=np.float32)
    tvec = np.array([0,0,0],dtype=np.float32)

    dist = np.array([0,0,0,0,0],dtype=np.float32)
    # return cv2.projectPoints(pt.T.reshape(-1,1,3),rvec,tvec,mat,dist)

    pts_2d = np.array([cv2.projectPoints(pt,rvec,tvec,mat,dist)[0] for pt in lines])

    ''' 
     Undistort points
    '''
    # undis_pts = cv2.undistortPoints(pts_2d.reshape((-1, 1, 2)), mat, dist).ravel()
    # return np.array([[pts_2d[i]*(1-undis_pts[i])] for i in range(len(pts_2d))])

    if ret_shape:
        return pts_2d.reshape(-1,2),calibrate()[2]
    return pts_2d.reshape(-1,2)

def manual_proj(rot,trans = np.array([0,0,0]),pts3d=np.array([[10,1000,1],[10,1000,10],[-10,1000,10],[-10,1000,100]],dtype=np.float32),intr = calibrate()[0],ret_shape = False):


    def non_hom_coords(pt3d):
        p1 = intr @ rot
        hom_res = p1 @ pt3d 
        coords = np.array([hom_res[0]/hom_res[2],hom_res[1]/hom_res[2]])
        return coords.ravel()

    def hom_coords(pt3d):
        if pt3d.shape[0] == 3:
            pt3d= pt3d.T; pt3d.shape = (3,1)

            nw = np.append(pt3d,np.array([[1]]),axis=0)
            pt3d = pt3d.T; pt3d.shape

        p1 = intr @ rot
        p = np.concatenate((p1,trans.T),axis=1)
        p_hom = np.concatenate((p,np.array([[0,0,0,1]])),axis=0)
        hom_res = p_hom @ nw
        coords = np.array([hom_res[0]/hom_res[2],hom_res[1]/hom_res[2]])
        return coords.ravel()

    if ret_shape:
        return np.array([non_hom_coords(pt) for pt in pts3d]),calibrate()[2]
    return np.array([non_hom_coords(pt) for pt in pts3d])


if __name__ == '__main__':
    print(project_pts().reshape(-1,2))
    # Test manual projection

    pitch = math.radians(60)
    yaw = math.radians(0)
    roll = math.radians(0)
    eX = np.array([[1,0,0],[0,math.cos( pitch),-math.sin( pitch)],[0,math.sin( pitch),math.cos( pitch)]]) 
    ey = np.array([[math.cos( yaw),0,math.sin( yaw)],[0,1,0],[-math.sin( yaw),0,math.cos( yaw)]])
    eZ = np.array([[math.cos( roll),-math.sin( roll),0],[math.sin( roll),math.cos( roll),0],[0,0,1]])
    rot = eX @ ey @ eZ
    print(manual_proj(rot,pts3d=np.array([[ -631.63070166, 1000,178.92571254],[637.88718405, 1000,178.92571254],[  975.63810363, 1000, 1219.70803391], [ -984.55732316,1000,1217.47690563] ])))


    # print(manual_proj(rot))