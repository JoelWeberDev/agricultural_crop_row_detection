"""
Author: Joel Weber
Date: 15/07/2023
Title: New previous lines
Description: A prev line module was developed before, but due to change of strategy and poor design in the previous one we are developing a relatively new one that implements some of 
the old ideas

Strategy: 
 1. Take group of detected lines along with the previous good lines
 2. Average the line groups to get a single represenation for each group
 3. Check if there are any gaps in the detected lines. If there are gaps, guess where the lines should be based on row spacing and the previous good lines
 4. Take all the potential groups and compare them to the previous lines to see which ones match the past in terms of location (horizontal shift minimization), spacing and missing lines
 5. Once the best group is found, update the previous lines with the new lines

 How is the comparison performed?
  Approach 1: Make a master function that has a weight for each of the 4 factors (horizontal shift, spacing, missing lines, and number of lines) and then returns a score. 
    How to ensure that we have a good representation of missing lines?
        This is a combination of using the past lines, what we have and the ideal row spacing 
        If there are potential inconsistencies then fill the gaps with the best match
        Keep the detected and predicted lines seperate. 
        Build a proximal search function to ensure that a line within a given area is indeed on the center of a row
    How to compare spacing?
    How do we want to shft the lines if there is a difference in spacing?
    Should we process the missing line or simply add them later? If we do process them how should we adjust the extent to which the affect the score?

  Approach 2: Disregard all the established groups and use the lines that lie nearest to the past lines.
  Approach 3: Don't bother with missing lines and only compare ones that are present. To get a weight for missed rows you can simply compare the ideal number of line with the actual number of lines



subtle nuances:
  - There may be some discrepancies between the spaced lines
  - The previous lines might not be fully representative of the actual lines. 
  - If there is a totally incorrect detection that has a higher likelihood of being propigated to the next frame
  - How do we also consider the locations where lines are not detected, but should be. 
  - Potential error make it difficult to determine where a row actually is. We also could have issue of detection on the side of the row rather than the center
  - Which direction to shift the set of lines if there are any missing lines or differences in spacing eg. on row matches well whereas another row is far out or shift it to about the center and equallize that differnece?


testing:
 - Test the line averaging
 - Test the placement of the missed lines
 - Test with complete real data
 - Use only one type of iterable for all the calculations
 - Test the distance fucntion 
 - Test the scoring to determine how much weight to give to each of the different factors (horizontal shift, spacing, missing lines)
 - Verify that the spacing is consistent with the ideal and the previous lines
 
Outline:
"""

from typing import Any
import cv2
import numpy as np  
import os, sys
import math
from icecream import ic
import matplotlib.pyplot as plt
from functools import reduce

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


# custom error for that is raise the inputted lines are averagaes and not the actual groups lines
class avg_lines_error(Exception):
    def __init__(self, message="The inputted lines are averages and not the actual groups lines"):
        self.message = message
        super().__init__(self.message)

class lines_already_averaged(Exception):
    def __init__(self, message="The inputted lines are already averages"):
        self.message = message
        super().__init__(self.message)

class process_prev_lines(object):
    def __doc__(self):
        return "This module will process the previous lines and determine the optimal lines for the next frame based on the previous lines"

    def __init__(self,avg=True,prev_lines=None):
        self.good_dtypes = [np.float32, np.float64, np.float16, np.float_, float,int, np.int32, np.int64, np.int16, np.int_]
        self.img = np.zeros((360,640))
        self.ap = ap()
        ic(self.ap.access("parameter path", universal=True),self.ap.access("row_num"))

        # self.spacing = self.ap.access("avg_spacing_pxls")
        # self.err_margin = self.ap.access("inlier_coeff")*self.spacing

        self.good_dtypes = [np.float32, np.float64, np.float16, np.float_, float,int]
        self.prev_lines = prev_lines 

        # self.last_lines = self._verify_inputs(last_lines) 
        # self.current_lines = self._verify_inputs(current_lines)
        self.avg = avg
        self.near_lines = False

    def _get_spacing(self,coeff=1):
        self.spacing = self.ap.access("avg_spacing_pxls")
        return self.spacing*coeff

    def _get_err_margin(self):
        self.err_margin = self.ap.access("inlier_coeff")*self._get_spacing()
        return self.err_margin

    def _get_row_num(self):
        self.row_num = self.ap.access("row_num")
        return self.row_num

    def _ensure_sorted(self, lines):
        if type(lines) in [list, tuple]:
            sorted_lines = sorted(lines, key=lambda x: x[1])
        elif type(lines) == np.ndarray:
            sorted_lines = lines[lines[:,1].argsort()]
        else:
            raise TypeError("The inputted lines must be either a list, tuple, or numpy array")

        for i in np.arange(len(sorted_lines)-1):
            if sorted_lines[i][1] > sorted_lines[i+1][1]:
                ic(sorted_lines)
                raise ValueError("array is not sorted")
        return sorted_lines
    
    def _make_unique(self, lines):
        if type(lines) in [list, tuple]:
            if len(lines) < 2:
                return lines
            unique_lines = list(set(lines))
        elif type(lines) == np.ndarray:
            if lines.shape[0] < 2:
                return lines
            unique_lines = np.unique(lines, axis=0)
        else:
            raise TypeError("The inputted lines must be either a list, tuple, or numpy array")

        return self._ensure_sorted(unique_lines)
        
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
            # else:
            #     raise TypeError("Invalid data type for lines")

        except IndexError:
            raise lines_already_averaged("The shape and dimesnionality of the lines is incorrect")

        except TypeError:
            raise avg_lines_error("Invalid data type for lines")

    # i is the number of elements left in the list
    # Input types: array: a slice of all the anchor intercepts from the last checked index to the end of the array 
    #              value: the current intercept of the line being checked
    #           i: the number of elements left in the array 
    def _find_nearest(self, array, value,i):
        idx = np.abs(array-value).argmin()
        if i + idx >= len(array):
            idx = len(array)-i
        return idx

    # this determines which lines are closest to the anchor lines
    def _min_dist(self, anchor, relative):
        # returns the indices of the relative lines that are closest to the anchor lines
        """
        Master shift function changes:
            Issue: If the lines are slighly misaligned sometime the system will prefer to perform a large shift. We would prefer having a small shift if it means keeping some lines closely aligned
            How to penalize large shifts and determine if it is best to keep some rows aligned
              1. If the misaglined row is gap filling line already then the penalization should be lower
              2. If the average of the diff  Multiply all the differneces together 
        """
        def search_min(arr, anch):
            get_prev = lambda v, p, i: self._find_nearest(anch[p:,1], v[1],i)+p
            prev_ind = 0
            diff = []
            inds = []
            for i, ln in enumerate(arr):
                prev_ind = get_prev(ln, prev_ind, len(arr)-i)   
                inds.append(prev_ind)
                # ic(ln, anch[prev_ind])
                diff.append(abs(ln[1]-anch[prev_ind][1]))
                prev_ind += 1
            return inds, diff
        
        f_inds, f_diff = search_min(relative, anchor)
        b_inds, b_diff = search_min(relative[::-1],anchor[::-1])

        # ic(f_inds, f_diff, b_inds, b_diff)
        if sum(f_diff) > sum(b_diff):
            b_inds = np.arange(len(anchor)-1,-1,-1)[b_inds].tolist()
            return b_inds[::-1], b_diff
        return f_inds, f_diff
    
    def _bin_search_min(self, anchor, relative, v_min, v_max):
        # ic(relative+v_min, relative+v_max)
        # returns true if the first is less than the second
        def compare_shift(s1,s2):
            assert type(s1) == list and type(s2) == list, "The inputted shift must be a list"
            assert len(s1) > 0, "The inputted shift must have at least one element"
            assert len(s1) == len(s2), "The inputted shifts must be the same length"

            # *Note this threshold may need to be a function of some more parameters if the muliplicaiton is occuring too soon
            variance_thresh = self._get_err_margin()
            # THe sums all the elements of the list if it is below the threshold otherwise it multiplies them since that will punish the large shifts more than the smaller one
            make_val = lambda s: sum(s) if sum(s)/len(s) <= variance_thresh else reduce(lambda x,y: x*y, s)
            val1,val2 = make_val(s1), make_val(s2)

            if val1 < val2:
                return True 
            return False
        mid = round((v_min+v_max)/2)
        min_res = self._min_dist(anchor, relative+v_min)
        max_res = self._min_dist(anchor, relative+v_max)
        if v_min >= v_max:
            assert min_res[0] == max_res[0], f"min_res {min_res} max_res {max_res} v_min {v_min} v_max {v_max} mid {mid} anchor {anchor} relative {relative} the search returned before convergence."
            return sum(max_res[1])/len(anchor), max_res[0], abs(v_max)
        
        if compare_shift(min_res[1],max_res[1]): 
            return self._bin_search_min(anchor, relative, v_min, mid-1)
        return self._bin_search_min(anchor, relative, mid+1, v_max)

    def _tally_missing_lines(self, prev, cur):
        missed = np.intersect1d(np.where(np.isnan(cur[:,0])) ,np.where(~np.isnan(prev[:,0])))
        absent = np.intersect1d(np.where(np.isnan(cur[:,0])) ,np.where(np.isnan(prev[:,0])))
        # print(np.where(np.isnan(cur[:,0])), np.where(~np.isnan(prev[:,0])), len(missed))
        # for i,(p,c) in enumerate(zip(prev, cur)):
        #     if np.isnan(c[0]):
        #         cur[i] = p

        return len(missed), len(absent)

    # Group 1 i
    def error_between_groups(self, prev, cur):
        # avg_diff = lambda a: np.mean(np.abs(np.diff(a)))
        try:
            self._verify_inputs(prev)
            # prev = avg_diff(prev_ints[:,1])
            prev_ints = self.calc_group_intervals(prev,avg=True)
        except (lines_already_averaged, avg_lines_error):
            # avg_prev = prev
            prev_ints = self.calc_group_intervals(prev)
            
        try:
            self._verify_inputs(cur)
            cur_ints = self.calc_group_intervals(cur,avg=True)
            # cur = avg_diff(cur_ints[:,1])
        except (lines_already_averaged, avg_lines_error):
            cur_ints = self.calc_group_intervals(cur)

        # greater than 0 if prev is greater than cur
        len_diff = len(prev_ints)-len(cur_ints)
        

        #  Here the score is implemented to account for the number of missing lines. The weight of a missing line could be somewhat subjective and thus we will add it to the adaptive parameters. Things like 
        # number of rows and the likelyhood of missing a line will contribute to this function.
        # This is somewhat arbitrary but 1/3 seems to be a good amout
        log_coeff = 1/3
        coeff = 1.2
        res = self._bin_search_min(prev_ints, cur_ints, self._get_spacing(-1/2), self._get_spacing(1/2)) if len_diff > 0 else self._bin_search_min(cur_ints, prev_ints, self._get_spacing(-1/2), self._get_spacing(1/2))

        # The goal is to establish a detection of each line at least once in sequence of images to determine ensure that the prev_ints has at least the correct amount of rows



        missed_rows, absent = self._tally_missing_lines(prev_ints[res[1]], cur_ints) if len_diff > 0 else self._tally_missing_lines(prev_ints, cur_ints[res[1]])
        # error = lambda rows : self._get_spacing()*math.log(rows+1, len(prev_ints))
        error = lambda rows : self._get_spacing()*math.sqrt(rows/len(prev_ints))*coeff
        score = (res[0]+res[2])+error(missed_rows)+(log_coeff)*error(absent) if missed_rows > 0 else (res[0]+res[2])

        # This will take two arrays of differing lengths and add the missing values from the shorter array to the longer array. These missing values are added as np.nan for the slope
        _get_not_res = lambda lg, r, sm: np.concatenate((np.array([[np.nan, lg[nr][1]] for nr in np.setxor1d(np.arange(len(lg)),r[1])]),sm))
        

        # Will fail if lines are detected to be in a too near proximity to each other
        if len(cur_ints) < self._get_row_num() and len_diff > 0:
            final_lines = _get_not_res(prev_ints, res, cur_ints)        
            for i, ln in enumerate(final_lines[:-1]):
                try: 
                    assert abs(final_lines[i][1] - final_lines[i+1][1]) > (self._get_spacing(3/4) - self._get_err_margin()), f"Overlapping lines final_lines {final_lines} cur_ints {cur_ints} prev_itns {prev_ints} res {res} xor {np.setxor1d(np.arange(len(prev_ints)),res[1])}"
                except AssertionError:
                    self.near_lines = True
                    res = self._bin_search_min(prev_ints, cur_ints, self._get_spacing(-1/2), self._get_spacing(1/2)) if len_diff > 0 else self._bin_search_min(cur_ints, prev_ints, self._get_spacing(-1/2), self._get_spacing(1/2))
                    ic(res)
                    assert False, "Lines are too near see error in terminal for more info"
        else:
            final_lines = cur_ints

        return score, missed_rows, final_lines ,res[2]

    """
    Lines data format:
        iterable of shape (n,n,2)
        they are in the form of gadient and intercept (intercepts is given as the y the x_value where y = image height) 
    """
    def calc_group_intervals(self, lines, **kwargs):

        avg = kwargs.get("avg", False)
        if avg:
            avg_lines = np.array([lc.avg_slope_lns(lns) for lns in lines])
        else: 
            avg_lines = np.array(lines)
        # sort lines

        # avg_lines = np.unique(avg_lines, axis=0)
        # avg_lines = avg_lines[np.argsort(avg_lines[:,1])]
        avg_lines = self._make_unique(avg_lines)

        # Difference calulator tailored to the avg_lines format
        diff = lambda i:  abs((avg_lines[i+1][1]-avg_lines[i][1])) 

        # Returns the numbers of rows missed for a given difference. *Note the rounding will determine if the gap is too narrow to be filled
        row_missed = lambda d: abs(round((d/self._get_spacing())))
        interval = lambda d: d/row_missed(d)
        # This will generate evenly spaced lines according to the difference between the nearest two lines 
        #  d: difference between lines or if on an edge specified otherwise
        #  e: edge of the image. Binary: 0 if not edge 1 if edge
        #  v: The start point of each interval
        # blanks = lambda d, e ,v: [np.array([np.nan, v+interval(d)*(i+1)]) for i in range(row_missed(d+e))]
        def blanks(d, e ,v, edge=True):
            rows_missed = row_missed(d+e)
            for i in range (rows_missed):
                if edge:
                    if d < 0:
                        yield np.array([np.nan, v-(self._get_spacing()*(i+1))])
                    else:
                        yield np.array([np.nan, v+(self._get_spacing()*(i+1))])
                else:
                    yield np.array([np.nan, v+interval(d)*(i+1)])

        possible_ints = []

        assert type(avg_lines) == np.ndarray, f"avg_lines is not a numpy array {avg_lines}"
        n_rows_missed = self._get_row_num()-avg_lines.shape[0]
        edge_candidates = []
        for i, ln in enumerate(avg_lines):
            if i == 0:
                if n_rows_missed > 0 and ln[1] > self._get_spacing()-self._get_err_margin():
                    ret = list(blanks(-ln[1],-self._get_err_margin(),ln[1]))
                    ret.reverse()
                    edge_candidates += (ret)

            # Remove the any values that are too near to eachother
            # last_int = possible_ints[-1] if len(possible_ints) > 0 else np.array([None,None])
            if len(possible_ints) > 0 and abs(ln[1]-possible_ints[-1][1]) < self._get_err_margin():
                last_int = possible_ints[-1] 
                possible_ints[-1] = np.average([possible_ints[-1],ln], axis=0) if last_int[0] != np.nan else [ln[0], (ln[1]+last_int[1])/2]

            else:
                assert len(ln) == 2, "The lines must be in the form of [slope, intercept]"
                possible_ints.append(ln)

            if i == len(avg_lines)-1:
                if n_rows_missed > 0 and ln[1] < self.img.shape[1]-(self._get_spacing()-self._get_err_margin()):
                    edge_candidates += list(blanks((self.img.shape[1]-ln[1]),self._get_err_margin(),ln[1]))

            elif diff(i) > self._get_spacing()+self._get_err_margin():
                possible_ints += list(blanks(diff(i),0,ln[1],False))

        # Determine which edges should be included in the possible intervals
        pos_diff = len(possible_ints) - self._get_row_num()
        if  pos_diff < 0 and len(edge_candidates) > 0:
            # Find the closest edges to the center of the image sort by huristic proximity to the center
            prev_len = len(edge_candidates)
            edge_candidates = sorted(edge_candidates, key=lambda x: abs(x[1]-self.img.shape[1]/2))[0:abs(pos_diff)]
            assert len(edge_candidates) in [abs(pos_diff), prev_len], "The number of edge candidates is not correct"

            if len(edge_candidates) > 1:
                edge_candidates = self._make_unique(np.array(edge_candidates)).tolist()
            possible_ints += edge_candidates

        # sort possible intervals
        possible_ints = np.array(sorted(possible_ints, key=lambda x: x[1]))
        assert possible_ints.shape[1] == 2, "The shape of the possible intervals is not correct"    

        return possible_ints

    def best_match(self, cur_groups):
        groups = sorted([self.error_between_groups(self.prev_lines, cur) for cur in cur_groups] , key=lambda x: x[0])
        return groups[0]



    
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



class testing(object):
    def __init__(self):
        from data_base_manager import data_base_manager as dbm 

        self.real_data = {
            "winter_wheat":dbm(data_name='winter_wheat.csv'),
            "corn":dbm(data_name='corn_test.csv'),
        }

        self.params_manager = ap()

    def test():


        # avg1 = [[ -3.86842491, 122.69957081],
        avg1 = [[
                # [ -5.16682097, 172.6018076 ],
                # [ -7.88389796, 231.35162428],
                # [-39.14285667, 295.87287783],
                [ 29.31514904, 344.54168174]]]
                # [ 11.00832752, 395.18935175],
                # [  6.47734778, 443.70262165],
                # [  5.00475157, 492.68410375]]

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
        ic(proc_prev.error_between_groups(avg2,avg1))

        # ic(proc_prev.calc_group_intervals(groups[1]))

        # avg_vals = np.average(np.array([np.average(np.array([proc_prev.error_between_groups(samp[0],samp[1])[1] for samp in gen_tests()]))for i in range(100)]))
        # for i, samp in enumerate(gen_tests()):
        #     print(proc_prev.error_between_groups(samp[0],samp[1])[2])

    def test_real_data(self, disp=False):
        data = self.real_data["corn"].read_data_base()
        self.params_manager.update("parameter path", "Adaptive_params\\tests\small_corn\\test.json", title="Current Settings", universal=True)

        # data = self.real_data["winter_wheat"].read_data_base()

        # Load the json file that corresponds to the data to ensure that correct spacing is achieved
        # groups = [alg_test(lns, origin=(0,360))[0] for lns in data]

        proc_prev = process_prev_lines(avg=True,prev_lines=data[0])

        for i, lns in enumerate(data[1:]):
            # Alg_test return a ordered list of all the line  group by the number of lines in each group
            prev_frame = alg_test(data[i], origin=(0,360))[0][0]
            proc_prev.prev_lines = prev_frame
            best = proc_prev.best_match(alg_test(lns, origin=(0,360))[0])
            if disp:
                self.display_res(best[2], np.zeros((360,640)))


    # The applies the lines to a numpy zerors array and shows it using matplotlib
    def display_res(self, lines, img=np.zeros((360,640,3))):
        assert len(lines[0]) == 2, "The lines are not in the correct format"

        if type(img) == np.ndarray:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB) 

        for line in lines:
            # convert gradient-intercept line into x1,y1,x2,y2 line
            if np.isnan(line[0]):
                x1,y1,x2,y2 = (int(line[1]),0,int(line[1]),img.shape[0])
                col = (255,0,255)
            else:
                x1,y1,x2,y2 = lc.bot_interc_ln_pts(line[0],line[1],img.shape)
                col = (255,0,0)
            cv2.line(img,(x1,y1),(x2,y2),col,2)
        
        plt.imshow(img)
        plt.show()
        print("showing")






if __name__ == "__main__":
    import system_operations as sys_op
    sys_op.system_reset()
    tests = testing()
    tests.test_real_data(disp=True)
    # load_test_data()
    # print("test

