"""
Author: Joel Weber
Title: Aggregate Lines
Overview: This is the final leg of the row detection. Where the optimal placement of the rows predicted is made.
"""

'''
outline:
 1. Calculate the intecpt of each line at the base value where the disparity is the greatest
 2. Calculate the optimal disparity and the error margin
 3. Iterate through every line and determine which ones fall in the descrete increment factors of the line under scrutiny
 4. Each line that lies within the near error range will be added and assimilated into the group of the scrutinized line
 5. Those lines that fall into the weak match will be summed and added to that group's score however they will be assessed later independantly

Future Developments:
 - Integrate the stero vision heat mapping to determine the average depth of the lowest 10 pixels
 - Use that average give a more precisce height value for the camera relative to the plants and the ground plane at the moment
'''


from icecream import ic
import sys, os
import numpy as np
import cv2
import math
import itertools

try:
    sys.path.append(os.path.abspath(os.path.join('.')))
    from Modules.Camera_Calib.Camera_Calibration import vanishing_point_calculator as vpc 
    from Adaptive_params.Adaptive_parameters import param_manager as ap
    from Modules.Camera_Calib.Chess_board_calib import manual_proj as threeD_to_twoD
    from Modules import line_calculations  as lc
except ModuleNotFoundError or ImportError:
    raise("Module not found at the correct path")



class ag_lines(object):
    def __init__(self,img,lines, **kwargs):
        # ic(lines, type(lines))
        self.img = img

        assert len(lines) > 1, "No lines were detected in the image"
        self.lines = self._censor_inp(lines) 

        # ic(self.lines, type(self.lines))
        self.conversion_table = {"m":1000,"cm":10,"km":1000000,"in":25.4,"ft":304.8,"yd":914.4,"mi":1609344, "mm":1}
        self.vpc = vpc()
        self._load_params()


    # This returns the x_value if the line at the provided y_val
    def _calc_intercept(self, ln, y_val = None):
        zero_int = self.img.shape[0] - (ln[0]*ln[1])
        if y_val:
            return (y_val-zero_int)/ln[0]
        else:
            ln[1]

    def _load_params(self):
        self.ap = ap()
        try: 
            self.spacing = self._cnv_unit(self.ap.access("spacing"),self.ap.access("unit"))
        except KeyError:
            raise("Invalid unit of measurement")

        self.spacing_px = self.ap.access("avg_spacing_pxls")
        # self.ln_err = self.ap.access("inlier_coefficient")
        self.ln_err = self.ap.access("inlier_coeff")
        self.neig_err = self.ap.access("neighbor_coefficient")


    def _cnv_unit(self, val, unit):
        return val * self.conversion_table[unit]

    """
    Takes an initail point on the line and the spacing distance and calculates the number of pixels between the two points
    Allow for a 5% error margin

    Test: Did the theoretical predictions match the acutal measured values? 
     - Yes, a test was preformed with the camera at 1000 mm above a level plane where the coner areas of the image were marked and measured for horizontal disparity
       the results were calculated to be within 5% of the theoretical values
    Note that the intercepts that are returned are the ones that have been processed and potentilly changed 
    """
    def _calc_row_disparity(self, pt, ui=False):        
        if ui:
            td_disparity = (td_pt := self.vpc.twoD_to_3D(pt)) + np.array([self.spacing,0,0])
            ic(td_disparity, td_pt)
            twoD_disp = threeD_to_twoD(np.array(self.vpc.extrinsic)[:3,:3], pts3d=td_disparity).ravel()
            delt_horz = int(abs(twoD_disp[0] - pt[0]))
            ic(delt_horz)
            return delt_horz
        else:
            return self.spacing_px

    def spacing_intervals(self, st_pt):
        st_pt,incpt_lns,srt_lns, disp,minmax= self._proc_lines(st_pt)
        # ic(minmax)
        sp_min,sp_max = st_pt[0] - math.ceil((st_pt[0] - minmax[0])/disp)*disp,st_pt[0] + math.ceil((minmax[1] - st_pt[0])/disp )*disp
        # ic(sp_min,sp_max)

        return np.array([[i, st_pt[1]] for i in np.arange(sp_min,sp_max+disp, disp)])

    def _proc_lines(self, st_pt):
        st_pt = self._censor_inp(st_pt).ravel()
        
        incpt_lns = np.array([[ln[0],self._calc_intercept(ln,st_pt[1])] for ln in self.lines])
        # ensure that the no lines are lost in the process of conversion to intercepts
        assert len(incpt_lns) == len(self.lines), "The number of lines and intercepts do not match"

        srt_lns = incpt_lns[incpt_lns[:,1].argsort()]
        # enusre that the lines are sorted in ascending order
        assert len(srt_lns) == len(self.lines), "The number of lines and sorted lines do not match"
        
        disp = self._calc_row_disparity(st_pt)
        # ic(disp)
        x_min,x_max = srt_lns[0,1], srt_lns[-1,1]

        return st_pt,incpt_lns,srt_lns, disp,(x_min,x_max)


    def calc_pts(self,st_pt=(0,720),ret_ranked_grps=False):
        # st_pt = self._censor_inp(st_pt).ravel()
        # incpt_lns = np.array([[ln[0],self._calc_intercept(ln,st_pt[1])] for ln in self.lines])
        # disp = self._calc_row_disparity(st_pt)

        st_pt,incpt_lns,srt_lns, disp,minmax = self._proc_lines(st_pt)

        assert len(srt_lns) > 1, "No lines provided please provide at least 1 group of lines to calc_pts"

        """
        Take both the inlier and talliable error margins and proform the binary search on them to determine how many lines are to be tallied. 
        Take the inliers and put them into a group. Then remove those inliers from the list of lines so that they do not get processed. 

        """
        def _tally_score(start, end, disp,lines,inc=True):
            sum = 0
            group = []
            iters = np.array([0])
            start = lines[0,1]
            while True:
                if inc and start > end:
                    break
                elif not inc and start < end:
                    break
                inliers = self._bin_search(lines[:,1],start,abs(disp*self.ln_err))

                # neighbours = self._bin_search(srt_lns[:,1],start,abs(disp*self.neig_err))

                sum += inliers[1]-inliers[0]

                inlier_iters = np.arange(inliers[0],inliers[1], dtype=int)
                inlier_vals = lines[inlier_iters]
                if inlier_iters.size > 0:
                    iters = np.unique(np.concatenate((iters,inlier_iters)))
                    group.append(inlier_vals.tolist())
                    # group += lines[inlier_iters].tolist()
                    # av_ln = np.average(inlier_vals,axis=0)
                    # av_ln = lc.avg_slope_lns(inlier_vals)
                    start = np.average(inlier_vals[:,1])
                start += disp
                
            # return sum,group,lines
            return sum,group,iters

        scored_grps = []
        scores = []
        max_score = (0,0)
        i = 0
        # for i,ln in enumerate(srt_lns):
        while len(srt_lns) > 0:
            ln = srt_lns[0]
            start = ln[1]
            x_min,x_max = srt_lns[0,1], srt_lns[-1,1]
            decrem, increm = _tally_score(start,x_max,disp, srt_lns,inc=True),  _tally_score(start,x_min,disp*-1,srt_lns,inc=False)

            scored_grp = increm[1] + decrem[1]
            tot_score = increm[0] + decrem[0]
            if tot_score > max_score[0]:
                max_score = (tot_score,i)
            if len(scored_grp) > 0:
                scored_grps.append(scored_grp)
                scores.append(tot_score)
                i += 1
            rem_vals = np.unique(np.concatenate((increm[2],decrem[2])),axis=0).astype(int)
            new = np.delete(srt_lns,rem_vals,axis=0)
            srt_lns = new[new[:,1].argsort()]



        try:
            res = list(itertools.chain(*scored_grps[max_score[1]]))
        except IndexError:
            ic(len(scored_grps))

            res = []

        if ret_ranked_grps:
            return scored_grps, scores,scored_grps[max_score[1]]


        return res,scored_grps[max_score[1]]



    """The input array must be a sorted one by the axis which you desire to search
    """        

    def _bin_search(self, arr, val,err): 
        
        ub,lb = round(val+err),round(val-err)
        if lb > arr[-1] or ub < arr[0]:
            return None,None
        high,low = len(arr)-1,0

        def low_search(low,high):
            if high-low < 2:
                # if arr[high] > ub:
                #     return None
                return high 
            mid = int((high+low)/2)
            if arr[mid] >= lb: 
                return low_search(low,mid)
            return low_search(mid,high)

        def high_search(low,high):
            if high-low < 2:
                # if arr[low] < lb:
                #     return None
                return low
            mid = int((high+low)/2)
            if arr[mid] <= ub:
                return high_search(mid,high)
            return high_search(low,mid)

        ret_low = low if lb < arr[low] else low_search(low,high)
        ret_high = high if ub > arr[high] else high_search(low,high)


        return ret_low, ret_high



    def disp_pred_pts(self,pts, pt_sz = 5, color = (255,0,0)):
        pts = self._censor_inp(pts)

        for pt in pts:
            cv2.circle(self.img, tuple(pt), pt_sz, color, -1)

    # def _slope_to_pts(self, ln, y_int=0):
    #     interc = self._calc_intercept(ln,y_int)
    #     pt1 = (int(y_int),int(ln[1]))
    #     reg_int = int((y_int-ln[1])/ln[0])
    #     pt2 = (reg_int, 0)
    #     return (pt1,pt2)

    def _get_pts(self,slope, intercept):
        int_nrm = -1*round((intercept*slope-self.img.shape[0])) 
        x1 = round(int_nrm/slope)
        y1 = 0
        x2 = round(intercept)
        y2 = self.img.shape[0]
        return [(x1,y1),(x2,y2)]

    def _censor_inp(self, iter):
        if type(iter) == list or type(iter) == tuple:
            iter = np.array(iter)
        if len(iter.shape) == 1:
            iter = np.array([iter])
        return iter
    
    def disp_pred_lines(self, lines):
        # lines = self._censor_inp(lines)
        ic("display")
        for gr in lines:
            for ln in gr:
                pts = self._get_pts(ln[0],ln[1])
                cv2.line(self.img, pts[0],pts[1] , (255,255,255), 2)
        return lines 

def test(lines = None, origin = (0,720)):        
    img = np.zeros((720,1280))
    if not lines:
        lines = [[ -3.09756098,  -8.        ],
                    [ -2.7826087 , -66.        ],
                    [ -3.07228916,  -8.        ],
                    [ -3.08536585,  -6.        ],
                    [ -6.4       , 329.        ],
                    [ -3.08045977, -10.        ],
                    [ -3.06896552, -12.        ],
                    [ -3.05747126, -14.        ],
                    [ -3.09756098,  -9.        ],
                    [ -3.09302326, -12.        ],
                    [ -6.38      , 330.        ],
                    [ -2.84444444, -66.        ],
                    [ -3.08139535, -14.        ],
                    [ -3.06097561,  -4.        ],
                    [ -3.08510638, -15.        ],
                    [ -3.0875    ,  -3.        ],
                    [ -2.85      , -67.        ],
                    [ -6.35      , 331.        ],
                    [ -2.83783784, -62.        ],
                    [ -3.07594937,  -2.        ]]
    # lines = lines_from_db()[0]
    ag = ag_lines(img, lines)

    groups = ag.calc_pts(origin, True)
    srt_scores = np.array(groups[1]).argsort()
    ic(groups[2])
    ranked_group = [groups[0][i] for i in srt_scores[::-1]]
    # ic(ranked_group)
    
    # pts, gr_lines = ag.calc_pts(origin, ret_gr_lines=False)
    # ic(groups[0:4])
    # ag.disp_pred_lines(gr_lines)
    # return (*ag.calc_pts(origin), ag)
    return ranked_group, ag


def lines_from_db():
    from data_base_manager import data_base_manager as dbm 
    winter_wheat = dbm(data_name='winter_wheat.csv')
    data = winter_wheat.read_data_base()

    # groups = [alg_test(lns, origin=(0,360))[1] for lns in winter_wheat.read_data_base()]
    return data


if __name__ == "__main__":
    # bad_lines = np.array([[
    #     [ -3.09756098,  -8.        ],
    #     [1, 2334],
    #     [[ -2.7826087 , -66.        ]]
    # ]])


    # pts = ag.spacing_intervals((100,720))
    # ag.disp_pred_pts(pts, color=(255,255,255))

    # pts, gr_lines, ag = test()
    test()
    
    # ag.disp_pred_lines(gr_lines)

    # cv2.imshow("pts",ag.img)
    # # cv2.imshow("blanc", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()