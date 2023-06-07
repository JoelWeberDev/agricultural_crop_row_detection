"""
Purpose: make an accessable module to preform the line calculations like point conversions, intercept calculations and turing slope lines back into points
"""

import numpy as np
import math
import cv2
from icecream import ic 

def cnvt_iterables(iter):
    if type(iter) != np.ndarray:
        return np.array(iter)
    return iter

def calc_slope(pt1,pt2):
    try:
        if pt2[1]- pt1[1] == 0:
            print(pt1,pt2)

        return (pt2[1]- pt1[1])/(pt2[0]-pt1[0])
    except ZeroDivisionError:
        return 10e10

def calc_intercept(slope,pt):
    return pt[1] - (slope*pt[0])

def calc_x_val(line,y):
    slope,intercept = line[0],line[1]
    try:
        return (y-intercept)/slope
    except ZeroDivisionError:
        return 10e10

def calc_pts(slope,intercept):
    return np.array([[0,intercept],[calc_x_val(slope,intercept,0),0]])

def calc_line(pt1,pt2):
    slope = calc_slope(pt1,pt2)
    intercept = calc_intercept(slope,pt1)
    return np.array([slope,intercept])

def get_vec(pt1,pt2):
    # Get the angle from the first point to the second
    # Get the slope of the line
    return np.array([pt2[0]-pt1[0],pt2[1]-pt1[1]])

def get_intersection(ln1,ln2):
    # get the slopes of the lines
    # get the intercepts of the lines
    # get the x value of the intersection
    # get the y value of the intersection
    # return the intersection
    slope1,intercept1 = ln1[0],ln1[1]
    slope2,intercept2 = ln2[0],ln2[1]
    x = (intercept2-intercept1)/(slope1-slope2)
    y = slope1*x + intercept1
    return np.array([x,y])

def get_distance(pt1,pt2):
    # get the vector from the first point to the second
    # get the magnitude of the vector
    # return the magnitude
    vec = get_vec(pt1,pt2)
    return np.linalg.norm(vec)

# finds the perependicular distance to the line from the point 
def get_shortest_distance(ln,pt):
    # get the slope of the line
    # get the slope of the line perpendicular to the line
    # get the intercept of the line perpendicular to the line
    # get the intercept of the line perpendicular to the line that goes through the point
    # get the intersection of the two lines
    # get the distance between the point and the intersection
    # return the distance
    slope = ln[0]
    perp_slope = -1/slope
    perp_intercept = calc_intercept(perp_slope,pt)
    perp_ln = np.array([perp_slope,perp_intercept])
    intersection = get_intersection(ln,perp_ln)
    return get_distance(pt,intersection)

def calc_extremes(ln,im_shape):
    top_ex = np.array([calc_x_val(ln,0),0])
    bot_ex = np.array([calc_x_val(ln,im_shape[0]),im_shape[0]])
    return np.array([top_ex,bot_ex])

def in_range(val,range_edges):
    try: 
        np_edges = cnvt_iterables(range_edges)
    except TypeError:
        raise TypeError("range_edges must be an iterable")

    np.sort(np_edges)
    if val > np_edges[1]:
        return False, np_edges[1]
    elif val < np_edges[0]:
        return False,np_edges[0]
    else:
        return True,val


def slope_to_theta(slope,deg=False):
    if deg:
        return math.degrees(math.atan(slope))
    ang = math.atan(slope)
    if ang < 0:
        ang += math.pi
    return ang



# This determines how to organize the points to ensure lines don't cross and which corner a point is optimally assigned to
class clockwise_sort(object):
    def __init__(self,pts, img_shape):
        ic(img_shape)
        self.pts = pts
        self.img_shape = img_shape
        self.center = np.mean(pts,axis=0)
        self.boundary_pts = self.get_boundary_pts()
        self.points_stack = stack_structure()
        self.corners_stack = stack_structure()
        self.correct_pts,self.good_lines = self.optimize_area(pts)
        self.ordered_pts = self.corner_distance()
        

    def optimize_area(self,pts=None, maximize=False):
        # Check all the points combinations with each other and determine which one has the largest area this can be done by checking which combination of lines produces the smallest perimeter
        # use the dfs seach to try all the combinations of points and use dp to eliminate repeats
        ret = None
        self.points_stack.push(self.branch(0,list(np.arange(1,len(pts)))))
        comp_per= 10e10
        if maximize:
            comp_per = 0
        for node in self.points_stack.stack:
            sum = 0
            for pair in node:
                sum += self.calc_length(pts[pair[0]],pts[pair[1]])
            if sum < comp_per and not maximize:
                comp_per = sum
                ret = np.array(node)
            elif sum > comp_per and maximize:
                comp_per = sum
                ret = np.array(node)
        #     ic(sum)
        # ic(ret,comp_per)
        lines = np.array([pts[pair[0]] for pair in ret]).tolist()
        # check if the lines are clockwise or counter clockwise

        # ic(lines)
        return ret,lines



    def calc_length(self,pt1,pt2):
        # ic(pt1,pt2)
        ret = math.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
        # ic(ret, pt1,pt2)
        return ret
        # return np.linalg.norm(pt2-pt1)

    def branch(self, prev, possible,combos=[]):
        if len(possible) == 0:
            ln = [prev,0]
            combos.append(ln)
            return combos
        for i in range(0,len(possible)):
            pos_copy, comb_copy = possible.copy(), combos.copy()
            ln = [prev,possible[i]]
            next_val = possible[i]
            pos_copy.remove(next_val) 
            comb_copy.append(ln)
            self.points_stack.push(self.branch(next_val, pos_copy,comb_copy))
            
    def get_boundary_pts(self):
        return np.array([[np.min(self.pts[:,0]),np.min(self.pts[:,1])],[np.max(self.pts[:,0]),np.max(self.pts[:,1])]])

    def corner_distance(self):
        # This is the minimization function of the assigning the correctly ordered points to the corners by minimizing the distance 
        # between the points and the corners
        self.corners = np.array([[0,self.img_shape[0]],[0,0],[self.img_shape[1],0],[self.img_shape[1],self.img_shape[0]]])

        # ic(self.corners)
        # np.flip(self.corners,axis=0)
        def run_check(corners,pts):
            min_val = 10e10
            ret = None
            for i in range(len(corners)): 
                total = 0 
                for a in range(len(corners)):
                    # print(corners[(a-i)%len(corners)],pts[a])
                    total += self.calc_length(corners[(a-i)%len(corners)],pts[a])
                    # print(self.good_lines[(i+a)%len(self.corners)],self.corners[a])
                if total < min_val: 
                    min_val = total
                    if i == 0:
                        ret = pts
                    else:
                        ret = pts[i:] + pts[:i]

            return ret,min_val
        ret1,ret2 = run_check(self.corners,self.good_lines),run_check(self.corners[::-1],self.good_lines)
        ret = ret1[0] if ret1[1] < ret2[1] else ret2[0][::-1]

        # self.test_frame(ret)
                
        return ret


    def test_frame(self,ret):
        blank_im = np.zeros(self.img_shape)
        ic(ret)

        for i in range(len(ret)):
            cv2.line(blank_im,(int(ret[i][0]),int(ret[i][1])),tuple(self.corners[i]),(255,255,255),2)
            cv2.line(blank_im,(int(ret[i][0]),int(ret[i][1])),(int(ret[i-1][0]),int(ret[i-1][1])),(255,255,0),2)
        import matplotlib.pyplot as plt
        plt.imshow(blank_im)
        plt.show()






class stack_structure(object):
    def __init__(self):
        self.stack = []

    def push(self,node):
        if type(node) != list:
            return
        if node not in self.stack:
            self.stack.insert(0,node)

    def pop(self):
        return self.stack.pop()
    


def clock_sort_test():
    import random
    pts = np.array([[ 2.08064516, 47.37096774], [30.46774194, 0.91935484], [39.5, 0.91935484], [3.37096774, 82.20967742]]) 
    # pts = np.zeros((4,2))
    # for i in range(4):
    #     pts[i] = np.array([random.randint(0,640),random.randint(0,360)])
    sorter = clockwise_sort(pts,(360,640))
    # sorter.optimize_area(sorter.pts)

if __name__ == "__main__":
    clock_sort_test()