"""
Author: Joel Weber
Date: 14/06/2023
Description: Module that applies only to videos or sequential sets of images to consider past lines for the line detection in the next frame
*Notes: This is ONLY for videos or sequential sets of images
**Warnings: This module can be dangerous to use since it may propigate errors to the next frame therefore it aims to consider the past lines, but it will diregard them if there
            is a large deviation from the current line
Dependencies: 
 Global: cv2, numpy, os, sys, icecream, datetime, re
 Local: Adaptive_params (gets the line deviation from past error margin and the threshold for an alternative set of lines to be used)

Module outline:
 Class object: consider_prev_lines
    all lines will be grouped and ranked
  - Inputs: Last lines, current lines , current frame
  - Outputs: New optimal lines
  - Functions:
   - Error between groups:
    This will determine the offset of the line intervals.
    Additional error will be incurred if there are lines absent in specific intervals where the other group contains them
   - determine_error between top groups:
    - This could either take the top n groups in both the sets of frames 
"""
import cv2
import numpy as np  
import os, sys
import math
from icecream import ic

# Local libraries:
try:
    from Adaptive_params.Adaptive_parameters import param_manager as ap
    from Modules import line_calculations as lc
    from aggregate_lines import  ag_lines ,test as alg_test
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join('.')))
    from Adaptive_params.Adaptive_parameters import param_manager as ap
    from Modules import line_calculations as lc
    from Algorithims_tst.aggregate_lines import  ag_lines ,test as alg_test


class process_prev_lines(object):
    def __doc__(self):
        return "This module will process the previous lines and determine the optimal lines for the next frame based on the previous lines"

    def __init__(self,avg=True,prev_lines=None):
        self.img = np.zeros((360,640))
        self.ap = ap()
        self.spacing = self.ap.access("avg_spacing_pxls")
        self.err_margin = self.ap.access("inlier_coeff")*self.spacing

        self.good_dtypes = [np.float32, np.float64, np.float16, np.float_, float,int]
        self.prev_lines = prev_lines 

        # self.last_lines = self._verify_inputs(last_lines) 
        # self.current_lines = self._verify_inputs(current_lines)

        self.avg = avg

    def display(self, lines):
        for ln in lines:
            int_nrm =int(-1*(ln[1]*ln[0]-self.img.shape[0])) 
            x1 = -1*int(int_nrm/ln[0])
            y1 = 0
            x2 = int(ln[1])
            y2 = self.img.shape[0]
            pt1, pt2 = (x1,y1),(x2,y2)

            cv2.line(self.img, pt1, pt2, (255,255,255), 1)
        cv2.imshow("img", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _verify_inputs(self, iter):
        try:
            if type(iter[0][0][0]) in self.good_dtypes:
                return iter
            else:
                raise TypeError("Invalid data type for lines")

        except IndexError:
            raise IndexError("The shape and dimesnionality of the lines is incorrect")

    # i is the number of elements left in the list
    def _find_nearest(self, array, value,i):
        idx = np.abs(array-value).argmin()
        if i + idx >= len(array):
            idx = len(array)-i
        return idx
    
    def _min_dist(self, anchor, relative):
        # ic(anchor, relative)

        def search_min(arr, anch):
            get_prev = lambda v, p, i: self._find_nearest(anch[p:,1], v[1],i)+p
            prev_ind = 0
            diff = 0
            inds = []
            for i, ln in enumerate(arr):
                prev_ind = get_prev(ln, prev_ind, len(arr)-i)   
                inds.append(prev_ind)
                # ic(ln, anch[prev_ind])
                diff += abs(ln[1]-anch[prev_ind][1])
                prev_ind += 1
            return inds, diff
        
        f_inds, f_diff = search_min(relative, anchor)
        b_inds, b_diff = search_min(relative[::-1],anchor[::-1])

        # ic(f_inds, f_diff, b_inds, b_diff)
        if f_diff > b_diff:
            return b_inds[::-1], b_diff
        return f_inds, f_diff
    
    def _bin_search_min(self, anchor, relative, v_min, v_max):
        # ic(relative+v_min, relative+v_max)
        mid = round((v_min+v_max)/2)
        min_res = self._min_dist(anchor, relative+v_min)
        max_res = self._min_dist(anchor, relative+v_max)
        if v_min >= v_max:
            return max_res[1]/len(anchor), max_res[0], abs(v_max)

        if min_res[1] < max_res[1]:
            return self._bin_search_min(anchor, relative, v_min, mid-1)
        return self._bin_search_min(anchor, relative, mid+1, v_max)

    def _tally_missing_lines(self, prev, cur):
        missed = np.intersect1d(np.where(np.isnan(cur[:,0])) ,np.where(~np.isnan(prev[:,0])))
        absent = np.intersect1d(np.where(np.isnan(cur[:,0])) ,np.where(np.isnan(prev[:,0])))
        # print(np.where(np.isnan(cur[:,0])), np.where(~np.isnan(prev[:,0])), len(missed))
        return len(missed), len(absent)

    # Group 1 i
    def error_between_groups(self, prev, cur):
        # avg_diff = lambda a: np.mean(np.abs(np.diff(a)))
        try:
            self._verify_inputs(prev)
            # prev = avg_diff(prev_ints[:,1])
            prev_ints = self.calc_group_intervals(prev,avg=True)
        except IndexError:
            # avg_prev = prev
            prev_ints = self.calc_group_intervals(prev)
            
        try:
            self._verify_inputs(cur)
            cur_ints = self.calc_group_intervals(cur,avg=True)
            # cur = avg_diff(cur_ints[:,1])
        except IndexError:
            cur_ints = self.calc_group_intervals(cur)

        len_diff = len(prev_ints)-len(cur_ints)
        # Case that the previous frame has more lines than the current frame and therefore the previous frame will be the anchor


        #  Here the score is implemented to account for the number of missing lines. The weight of a missing line could be somewhat subjective and thus we will add it to the adaptive parameters. Things like 
        # number of rows and the likelyhood of missing a line will contribute to this function.
        # This is somewhat arbitrary but 1/3 seems to be a good amout
        log_coeff = 1/3
        coeff = 1.2
        res = self._bin_search_min(prev_ints, cur_ints, self.spacing*-1, self.spacing) if len_diff > 0 else self._bin_search_min(cur_ints, prev_ints, self.spacing*-1, self.spacing)
        missed_rows, absent = self._tally_missing_lines(prev_ints[res[1]], cur_ints) if len_diff > 0 else self._tally_missing_lines(prev_ints, cur_ints[res[1]])
        # error = lambda rows : self.spacing*math.log(rows+1, len(prev_ints))
        error = lambda rows : self.spacing*math.sqrt(rows/len(prev_ints))*coeff
        score = (res[0]+res[2])+error(missed_rows)+(log_coeff)*error(absent) if missed_rows > 0 else (res[0]+res[2])
        
        return score, missed_rows, res[2], cur_ints

    """
    Lines data format:
        iterable of shape (n,n,2)
        they are in the form of gadient and intercept (intercepts is given as the y the x_value where y = image height) 
    """
    def calc_group_intervals(self, lines, edges=True,**kwargs):
        avg = kwargs.get("avg", False)
        if avg:
            avg_lines = np.array([lc.avg_slope_lns(lns) for lns in lines])
        else: 
            avg_lines = np.array(lines)
        # sort lines
        avg_lines = avg_lines[np.argsort(avg_lines[:,1])]
        diff = lambda i:  abs((avg_lines[i+1][1]-avg_lines[i][1])) 
        row_missed = lambda d: abs(round((d/self.spacing)))-1
        interval = lambda d: d/(row_missed(d)+1)
        blanks = lambda d ,v: [np.array([np.nan, v+interval(d)*(i+1)]) for i in range(row_missed(d))]

        possible_ints = []
        for i, ln in enumerate(avg_lines):

            if i == 0 and edges:
                if ln[1] > self.spacing+self.err_margin:
                    ret = blanks(-ln[1],ln[1])
                    ret.reverse()
                    possible_ints += ret

            # Remove the any values that are too near to eachother
            last_int = possible_ints[-1] if len(possible_ints) > 0 else np.array([None,None])
            if last_int[1] != None and abs(ln[1]-last_int[1]) < self.err_margin:
                possible_ints[-1] = np.average([possible_ints[-1],ln], axis=0) if last_int[0] != np.nan else [ln[0], (ln[1]+last_int[1])/2]
            else:
                possible_ints.append(ln)

            if i == len(avg_lines)-1:
                if edges and ln[1] < self.img.shape[1]-(self.spacing+self.err_margin):
                    possible_ints += blanks((self.img.shape[1]-ln[1]),ln[1])

            elif diff(i) > self.spacing:
                possible_ints += blanks(diff(i),ln[1])

        # ic(possible_ints)                    

        return np.array(possible_ints)

    def best_match(self, cur_groups):
        # ic(self.calc_group_intervals(prev), [self.calc_group_intervals(cur) for cur in cur_groups])
        # ic(self.error_between_groups(prev, cur_groups[0]))
        groups = sorted([self.error_between_groups(self.prev_lines, cur) for cur in cur_groups] , key=lambda x: x[0])
        # self.prev_lines = groups[0][3]
        return groups[0]
        # return [self.error_between_groups(prev, cur) for cur in cur_groups]



    
def gen_tests(n_tests=100, ln_range=(6,12), img_size=(360,640), spacing=58, space_err=5 ):
    gen_err = lambda: np.random.randint(-space_err,space_err)
    gen_sample = lambda start, end, spacing: [[0, a + gen_err()] for a in range(start, end, spacing)] 
    def params():
        spacing = np.random.randint(45,65)
        n_lines = np.random.randint(ln_range[0],ln_range[1])
        err = np.random.randint(-space_err,space_err)
        start = round(img_size[1]/2 - (spacing*n_lines)/2-err)
        end = round(img_size[1]/2 + (spacing*n_lines)/2+err)
        return start, end, spacing
    for i in range(n_tests):
        yield [gen_sample(*params()) for i in range(2)]




def test():


    avg1 = [[ -3.86842491, 122.69957081],
    # avg1 = [[
            [ -5.16682097, 172.6018076 ],
            [ -7.88389796, 231.35162428],
            [-39.14285667, 295.87287783],
            [ 29.31514904, 344.54168174],
            [ 11.00832752, 395.18935175],
            [  6.47734778, 443.70262165],
            [  5.00475157, 492.68410375]]

    avg2 = [[  3.92631901, 522.61744617],
            [  4.68310404, 462.62624103],
            # [ 12.80478954, 392.54410897],
            # [ 37.55208302, 344.38260035],
            [ -7.38011602, 223.78373761],
            [ -5.44186068, 177.15384359],
            [  3.92631901, 522.61744617]]

    proc_prev = process_prev_lines([avg1],[avg2])
    # ic(proc_prev.calc_group_intervals(avg1))
    # ic(proc_prev.calc_group_intervals(avg2))
    ic(proc_prev.error_between_groups(avg1,avg2))

    # ic(proc_prev.calc_group_intervals(groups[1]))

    ic.disable()
    # avg_vals = np.average(np.array([np.average(np.array([proc_prev.error_between_groups(samp[0],samp[1])[1] for samp in gen_tests()]))for i in range(100)]))
    # for i, samp in enumerate(gen_tests()):
    #     print(proc_prev.error_between_groups(samp[0],samp[1])[2])

def test_real_data():
    ic.disable()
    from data_base_manager import data_base_manager as dbm 
    winter_wheat = dbm(data_name='winter_wheat.csv')
    data = winter_wheat.read_data_base()

    groups = [alg_test(lns, origin=(0,360))[0] for lns in winter_wheat.read_data_base()]
    proc_prev = process_prev_lines(avg=True,prev_lines=data[0])

    for i, lns in enumerate(data[1:]):
        # Alg_test return a ordered list of all the line  group by the number of lines in each group
        prev_frame = alg_test(data[i], origin=(0,360))[0][0]
        proc_prev.prev_lines = prev_frame
        best = proc_prev.best_match(alg_test(lns, origin=(0,360))[0])
        print(best)
        # scores = proc_prev.best_match(prev_frame, alg_test(lns, origin=(0,360))[0])

        # scores.sort(key=lambda x: x[0])
        # print(scores)


if __name__ == "__main__":
    # test()
    test_real_data()
    # load_test_data()
    # print("test

