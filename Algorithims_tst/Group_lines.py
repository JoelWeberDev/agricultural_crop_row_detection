"""
Author: Joel Weber
Title: Group_lines
Overview: Here we average the lines that are similar and proximal inot a single line
Visions: 
 - Eventually this module should be written in c++ for speed and efficency
 - Tie this in with the adaptive parameters module
 - 
"""


import numpy as np
from icecream import ic
import sys,os

try:
    sys.path.append(os.path.abspath(os.path.join('.')))
    from Adaptive_params.Adaptive_parameters import param_manager as ap
except ModuleNotFoundError or ImportError as e:
    print(e)
    from Adaptive_params.Adaptive_parameters import param_manager as ap



class group_lns(object):
    def __init__(self):
        self.row_num = 0
        self.groups = []
        self.ap = ap()
    ### Integrate into new module
    # Grouer planning: 
    # All the lines being inputted have been passed through the slope filter
    # 1. Calcualte the intercepts of the lines
    # 2. Sort the intercepts
    # 3. Iterate through the intercepts and group them by agjacentcy and slope.
    # 4. If a line does not fit within an existing group then generate a new group

    # The line input is in the format of a numpy array

    # How to employ a multivariable grouping algorithm
    # 1. Create a list of all the variables that will be used to group the lines
    # 2. Iterate through the list and group the lines by the first variable
    # 3. Iterate through the groups of the first variable and group them by the second variable


    # Problem: The array is not being properly sorted thus we need to fix that next.
    
    def groupLines(self,lines,img_shape = None):
        # ic(len(lines))
        if img_shape == None:
            img_shape = self.ap.access("im_dims")
        
        int_thresh = self.ap.access("intercept_perc")*img_shape[0]
        ic(int_thresh)
        slp_thresh = self.ap.access("slope_perc")

        self.groups = []
        self.row_num = 0
        # Sort the lines by intercept
        srt_lns = lines[lines[:,1].argsort()]
        split_ind = []
        prev_val = srt_lns[0,1]
        # *** What do we need to know to determine if it is breaking correctly?
        # - The values that it breaks at
        # - What does np.split actially do?
        # - 
        for i,ln in enumerate(srt_lns):
            if i == 0:
                continue
            if abs(ln[1] - prev_val) > int_thresh: # 
                split_ind.append(i)
            prev_val = ln[1]
        grps = np.split(srt_lns,split_ind)


        # return srt_lns 
        # filter the lines by slope

        # ** Potential problem: The slope filter must be adaptive as the diffence between slopes becomes more radical as they get steeper
        # Generate a function form the camera calculations that will determine the upper and lower range that the slopes should be within 
        # The slope filter must be made adaptive becaue slopes like 50 abs - 50 are actually very similar whereas something like 0.5 and -0.5 are very different

        # for gr1 in grps:
        #     if gr1.size == 0: 
        #         continue
        #     # sort gr1 by slope 
        #     srt_gr1 = gr1[gr1[:,0].argsort()]
        #     self.groups.append([list(srt_gr1[0])])
        #     prev_sl = gr1[0,0]
        #     self.row_num += 1
        #     for i, ln in enumerate(srt_gr1):
        #         if i == 0:
        #             continue
        #         if abs(ln[0] / prev_sl)-1 > slp_thresh:
        #             self.groups.append([ln])
        #             self.row_num += 1
        #         else:
        #             self.groups[self.row_num-1].append(list(ln))
        #         self.prev_sl= ln[0]

        # ic(self.groups)
        # return self.groups        
        return grps
    



if __name__ ==  "__main__":
    gr = group_lns()
    lines = np.array([[ -3.09756098,  -8.        ],
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
                  [ -3.07594937,  -2.        ]])
    ic(gr.groupLines(lines))

        