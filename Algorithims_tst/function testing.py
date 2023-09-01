# fucntion testing
import numpy as np
import math

def numpy_make_unique(arr):
    return np.unique(arr,axis=0)

def numpy_sort(arr, ax=0):
    return arr[arr[:,ax].argsort()]

    
def test():
    data = np.array([[         np.nan, 526.75675676],
            [ -8.12058804, 200.        ],
            [         np.nan, 420.        ]])
    
    print(numpy_make_unique(numpy_sort(data,ax=1)))


class apply_function(object):
    def __init__(self, function, *args):
        self.function = function
        self.args = args
        self.result = self.solve()
    def solve(self):
        return self.function(*self.args)

def test2():
    def add(a,b,c):
        return a+b+c

    def quadratic(a,b,c):
        return (-b + math.sqrt(b**2 - 4*a*c))/(2*a)

    def tensor_mult(a,b):

        assert a.shape == b.shape
        return np.tensordot(a,b,axes=0)

    adder = apply_function(add,1,2,3)
    quad = apply_function(quadratic,1,2,-3)
    tensor = apply_function(tensor_mult,np.array([1,2,3]),np.array([1,2,3]))

    funcs = [adder,quad,tensor]
    resuls = [i.result for i in funcs]
    print(resuls)

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



if __name__ == "__main__":
    # test2()
    # test the diagonal slicing
    dig_slicing = test_diagonal_slicing(None,None)
    # dig_slicing.test_slicing()
    dig_slicing.test_values_between()