'''
Author: Joel Weber
Title: Image Annotation
Overview: This is the module that will allow the user to annotate the image that will be used for
 verification that the theoretical values are correct

1. Display an image and take to the user 
2. Allow the user to make a bounding shape around the rows
 - To make this a nice interface develop the following features:
  - provide the following instructions:
   Draw a bounding box around the rows by placing a point at each corner of the row. When you draw
   the box ensure that most of the row has been surrounded, but if there is a leaf or protursion 
   that extends out of the row please exclude that. Try to draw boxes around every row that extends
   at least half the lenght of the screen. 

  - Undo and redo and escape
  - Click to make point and display line from last point to the mouse position
  - Allow 4 points 
  - Provide a confirm and verification button
  - Allow the user to at any time make another calibration image

  How to comapre the annotated with the theoretical?
   Processing functions:
    Green points:
     Parsing the diagonal:
      Since the lines are not aligned with the rows and columns we must parse the diagonal to process the row

Future extensions:
 - Use the model to take guess of where the lines should be and the user can correct it
 

     
'''

import numpy as np
import warnings
import math
from icecream import ic
import sys,os
# Mat dependecies
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.widgets import Button

try:
    from Image_Processor import apply_funcs as pre_process 
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join('.')))
    from Modules.Image_Processor import apply_funcs as pre_process 

from Modules.Camera_Calib.Chess_board_calib import manual_proj as threeD_to_twoD
from Modules.Camera_Calib.Camera_Calibration import vanishing_point_calculator as vpc 
# import json
from Modules.json_interaction import load_json

loaded_cam = load_json("C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Cascade_Classifier/Modules/Camera_Calib/Cam_specs_dict.json",dict_data=True)

vpc = vpc()

import Modules.line_calculations as lc


#  pylint: disable=E1101

class image_interaction(object):
    def __init__(self,img):
        # self.img = img

        self.img= pre_process(img,des=["resize"])
        self.img_shape = img.shape

        self.mouse_pos = (0,0)
        self.points = np.ones((5,2))
        self.cur_pt = 0

        self.annotater = disp_shape()

        self.buttons = {}

    def prep_plot(self):
        self.figure, self.ax = plt.subplots(1,1)
        plt.imshow(self.img),plt.title('Input')
        self._user_buttons()
        plt.connect('motion_notify_event', self.on_move), plt.connect('button_press_event', self.on_click)
        plt.show()

    def reset_pts(self):
        self.cur_pt = 0
        self.points = np.ones((5,2))

    def on_move(self,event):
        if event.inaxes:
            self.mouse_pos = (event.xdata, event.ydata)
            if self.cur_pt != 0:
                ic(self.cur_pt,self.points[self.cur_pt-1])
                hover_pts = np.array([list(self.points[self.cur_pt-1]),self.mouse_pos,[1.0,1.0]])
                self.annotater.shape_manager(event,hover_pts,self.cur_pt,hover=True)

    def on_click(self,event):
        if event.button is MouseButton.LEFT:
            if event.inaxes is self.ax:
                print("Left click")
                if self.mouse_pos[0] > self.img.shape[1]-1:
                    self.mouse_pos[0] = self.img.shape[1]-1 # type: ignore
                elif self.mouse_pos[1] > self.img.shape[0]-1:
                    self.mouse_pos[1] = self.img.shape[0]-1 # type: ignore
                self.points[self.cur_pt] = self.mouse_pos 
                self.cur_pt += 1
                self.annotater.shape_manager(event,self.points,self.cur_pt)
                if self.cur_pt == 4:
                    self.reset_pts()


    def on_confirm(self,event):
        print("Confirm clicked.")
        plt.close(event.canvas.figure)
        if self.cur_pt > 0:
            warnings.warn("The user has not finished annotating the image. The current points will be discarded")     
        self.reset_pts()    
        self.annotations = self.annotater.annotations

    def on_undo(self,event):
        print("Undo clicked.")
        if self.cur_pt > 0:
            self.points[self.cur_pt] = [1.0,1.0]
            self.cur_pt -= 1
            if self.annotater.temp != []:
                self.annotater.remove_patch()
        elif len(self.annotater.temp) > 0: 
            ic(self.annotater.annotations)
            self.points = np.concatenate(self.annotater.annotations[-1][:3],np.ones((2,2))) # type: ignore
            ic(self.points)
            self.annotater.temp = self.points[:3] 
            self.annotater.remove_patch(self.annotater.annotations[-1][-1])

    def on_redo(self,event):
        print("Redo clicked.")

    def on_reset(self,event):
        print("Reset clicked.")
        self.reset_pts()
        for patch in self.ax.patches:
            patch.remove()
        for pts in self.ax.lines:
            pts.remove()
        fig = plt.gcf()
        fig.canvas.draw()
        self.annotater.reset()
        
    def _user_buttons(self):
        self.button_axs = {"confirm":[plt.axes([0.81, 0.05, 0.1, 0.075]),self.on_confirm], # type: ignore
            "undo":[plt.axes([0.7, 0.05, 0.1, 0.075]),self.on_undo], # type: ignore # type: ignore
            "redo":[plt.axes([0.59, 0.05, 0.1, 0.075]), self.on_redo], # type: ignore
            "reset":[plt.axes([0.48, 0.05, 0.1, 0.075]),self.on_reset]} # type: ignore

        for key,ax in self.button_axs.items():
            self._make_button(key,ax)

    def _make_button(self,key,ax):
        button = Button(ax[0], key, color='lightgoldenrodyellow', hovercolor='0.975')
        button.on_clicked(ax[1])
        self.buttons[key] = button

class disp_shape(object):
    def __init__(self):
        self.reset()
        # self.temp = []
        # self.annotations = []
        # self.conf_patches = []
        # self.hover = None

    def get_patch(self,verticies,code):
        path = Path(verticies,code)
        patch = PathPatch(path, facecolor='none', edgecolor='red', lw=2)
        return patch

    def _make_point(self,point,plot):
            pts = point.ravel()
            plot.plot(pts[0],pts[1], '.', markersize=5, alpha=0.4, color='red', zorder=10)
    
    def _make_line(self,line,plot,final=False,hover=False):
            code = [Path.MOVETO] + [Path.LINETO]*(len(line)-2) + [Path.CLOSEPOLY]
            verticies = self.type_handler(line,(list,tuple,int,))

            patch = self.get_patch(verticies,code)
            if final:
                self.annotations.append(line[:4])
                for p in self.temp:
                    p.remove()
                self.conf_patches.append(patch)
            elif hover:
                if self.hover:
                    self.hover.remove()
                self.hover = patch

            plot.add_patch(patch)

    def remove_patch(self,patch=None):
        if patch is None:
            self.temp[-1].remove()
        else:
            patch.remove()

    def reset(self):
        self.temp = []
        self.annotations = []
        self.conf_patches = []
        self.hover = None

            
    # The shape should always be numpy array of points that will be connected by direct lines
    def shape_manager(self,event,points,cur_pt,hover=False):
        plot = event.canvas.figure.axes[0]
        if hover:
            self._make_line(points,plot,hover=hover)
        elif cur_pt == 1:
            self._make_point(points[:cur_pt],plot)
        elif cur_pt == 4:
            self._make_line(points[:cur_pt+1],plot,final=True)
        else:
            self._make_line(points[cur_pt-2:cur_pt+1],plot)
        event.canvas.draw()

    # Error handling and type checking
    def type_handler(self,val,dtype):
        def change_iterable(val,depth):

            if depth == len(dtype)-1:
                return [dtype[depth](v) for v in val]
            else:
                return dtype[depth](dtype[depth](change_iterable(v,depth+1)) for v in val)



        if len(dtype) == 1:
            return dtype[0](val)
        else:
            return dtype[0](change_iterable(val,1))

def write_annotations(annotations):
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('.')))
    from Adaptive_params.Adaptive_parameters import param_manager as pm

    pm = pm()
    ic(annotations)
    # pm.update("calc_vals",annotations,"calc_vals")
    for key,annot in annotations.items():
        ic(key,annot)
        pm.update(key, annot,"calc_vals")

    return pm.path


def type_handle_test(class_obj):
    tests = [
        # [[1,2,3],(tuple,int)],
        [np.array([np.array([1,2,3.3]),np.array([1,2,3.3])]),(tuple,tuple,int,)],

    ]
    for test in tests:
        ic(class_obj.type_handler(test[0],test[1]))


class proc_annotations(object):
    def __init__(self,img,annotations):
        self.annotations = np.array(annotations)
        self.img = pre_process(img,des=["resize"])
        vp_key = 'vanishing_point{}x{}'.format(self.img.shape[0],self.img.shape[1])
        self.vp = loaded_cam.find_key(vp_key,True)
        self.disp_im = np.copy(self.img)
        self.make_mask()
        self.lr_mask()
        self.annot_info = {}

        self.json_info = {"annotations": self._make_list(annotations) ,"vp":self._make_list(self.vp)}
        self.find_rows()
        self.add_json()

    def _make_list(self, iter):
        iterable_types = (list,tuple,np.ndarray)

        if not type(iter) in iterable_types:
            raise TypeError("Input must be an iterable type")
        if  type(iter[0]) in iterable_types:
            return [self._make_list(it) for it in iter]
        else: 
            ret = iter.tolist() if type(iter) == np.ndarray else list(iter)
            return ret
        
        
    def make_mask(self):
        self.mask = pre_process(self.img,des=["kernel","mask"])

    def lr_mask(self):
        bin_im = np.where(self.mask > 0, 1, 0)
        self.lr_mat = np.cumsum(bin_im,axis=1)

    """
    This function will outline the following properties of the annotated image
     - Number of rows
     - Number of green pixels in each row
     - Center of each row
     - The width of each row at the top and bottom(pixels)
     - The average slope of each row (pixels)
     - The width of each row at the top and bottom(mm)
    """
    def find_rows(self,show=False):
        self.annot_info["row_num"] = 0
        self.annot_info["spacing_pxls"] = []
        self.annot_info["spacing_mm"] = []
        self.annot_info["annotated_green"] = 0
        self.annot_info["all_annot_pts"] = 0

        bot_width_pxl, top_width_pxl, bot_width_mm, top_width_mm = 0,0,0,0

        for annot in self.annotations:
            i = self.annot_info["row_num"]
            res,good_ln =  self.proc_row(annot) 
            self.annot_info["annotated_green"] += res["green_pts"]
            self.annot_info["all_annot_pts"] += res["tot_pts"]
            if good_ln:
                self.annot_info["row {}".format(i)] = res
                if i != 0:
                    self.annot_info["spacing_mm"].append(self._calc_row_disparity(res["bot_center"],self.annot_info["row {}".format(i-1)]["bot_center"]))
                    self.annot_info["spacing_pxls"].append(abs(res["bot_center"][0]-self.annot_info["row {}".format(i-1)]["bot_center"][0]))

                bot_width_pxl += res["bot_width_pxls"]
                top_width_pxl += res["top_width_pxls"]
                bot_width_mm += res["bot_width_mm"]
                top_width_mm += res["bot_width_pxls"] 
                self.annot_info["row_num"] += 1

        self.annot_info["avg_bot_width_pxl"] = bot_width_pxl/self.annot_info["row_num"]
        self.annot_info["avg_top_width_pxl"] = top_width_pxl/self.annot_info["row_num"]
        self.annot_info["avg_bot_width_mm"] = bot_width_mm/self.annot_info["row_num"]
        self.annot_info["avg_top_width_mm"] = top_width_mm/self.annot_info["row_num"]
        if len(self.annot_info["spacing_mm"]) > 0:
            self.annot_info["avg_spacing_mm"] = int(np.average(self.annot_info["spacing_mm"]))
            self.annot_info["avg_spacing_pxls"] = int(np.average(self.annot_info["spacing_pxls"]))
        else:
            raise Exception("Please define more than one row or define that only one row is present in the settings page")
        # self.annot_info["avg_spacing_mm"] = int(np.average(self.annot_info["spacing_mm"]))
        # self.annot_info["avg_spacing_pxls"] = int(np.average(self.annot_info["spacing_pxls"]))
        self.annot_info["all_green"] = np.sum(self.mask == 255)
            
        ic(self.annot_info)

        if show:
            plt.subplot(1,1,1)
            plt.imshow(self.disp_im)
            plt.show()

    def add_json(self):
        for key,val in self.annot_info.items():
            if "row " not in key:
                self.json_info[key] = val

    def proc_row(self, annot):
        print(annot)
        # determine the horizontal compontents in the annotation 
        cur_info = {}
        ln_slps = []
        # Format the points in an order that begins from the bottom left and goes clockwise
        # Clockwise point sort
        clock_srt = lc.clockwise_sort(annot,self.img.shape)
        srt_pts = clock_srt.ordered_pts

        for i,pt in enumerate(srt_pts): # type: ignore
            slp = lc.calc_slope(pt,srt_pts[i-1])
            ln_slps.append(slp)
            ic(slp)
         
        sr = self._check_lns_with_vp(srt_pts)
        # horz_lns = np.array([[srt_pts[sr[1]],srt_pts[sr[2]]],
        #                     [srt_pts[sr[3]],srt_pts[sr[0]]]])
        vert_lns = np.array([[srt_pts[sr[0]],srt_pts[sr[1]]],
                            [srt_pts[sr[2]],srt_pts[sr[3]]]])
        self.v_lines = np.array([lc.calc_line(ln[0],ln[1]) for ln in vert_lns])
        v_ext_pts = np.array([lc.calc_extremes(ln,self.img.shape) for ln in self.v_lines])
        h_ext_pts = np.array([[v_ext_pts[0][0],v_ext_pts[1][0]],[v_ext_pts[0][1],v_ext_pts[1][1]]])
        ic(h_ext_pts)
        
        
        # preform a check to see if the verical lines proceed towards the vanishing point

        self.vert_lns = vert_lns


        # ic(np.array(h_ext_pts[1]).T[:][:2], h_ext_pts)
        cur_info["green_pts"],cur_info["tot_pts"] = self._horz_sum()

        # 5% is the distance that the lines can intercept outside the frame to be considered viable
        in_marg = 0.05
        fr_range = (-1*self.img.shape[1]*in_marg,self.img.shape[1]*(1+in_marg))
        ic(lc.in_range(h_ext_pts[1][0][0],fr_range)[0] ,lc.in_range(h_ext_pts[1][1][0],fr_range)[0])
        if not (lc.in_range(h_ext_pts[1][0][0],fr_range)[0] and lc.in_range(h_ext_pts[1][1][0],fr_range)[0]):
            ic("bad line",h_ext_pts)
            return cur_info, False
        

        # set the calculated values
        cur_info["bot_center"] = np.average(np.array(h_ext_pts[1]).T[:][:2], axis=1)
        cur_info["top_center"] = np.average(np.array(h_ext_pts[0]).T[:][:2], axis=1)

        inv_sl = np.array([1/ln[0] for ln in self.v_lines])

        cur_info["avg_slope"] = 1/np.average(inv_sl) 

        cur_info["bot_width_pxls"] = abs(h_ext_pts[1][0][0]-h_ext_pts[1][1][0])
        cur_info["top_width_pxls"] = abs(h_ext_pts[0][0][0]-h_ext_pts[0][1][0])
        cur_info["bot_width_mm"] = self._calc_row_disparity(h_ext_pts[1][0],h_ext_pts[1][1])
        cur_info["top_width_mm"] = self._calc_row_disparity(h_ext_pts[0][0],h_ext_pts[0][1])
                

        # ic(cur_info)

        return cur_info,True
    
    # if retruns true then the lines are in the correct order
    def _check_lns_with_vp(self,lns,slopes=False):
        even_sum,odd_sum = 0,0
        for i,ln in enumerate(lns):
            if i == len(lns)-1:
                i = -1
            line = ln if slopes else lc.calc_line(ln,lns[i+1])
            dist = lc.get_shortest_distance(line,self.vp)
            dist = dist if not math.isnan(dist) else 10e10
            if i % 2 == 0:
                even_sum += dist
            else:   
                odd_sum += dist
        if even_sum < odd_sum:
            return np.arange(len(lns))
        else:
            print("reversed",lns)
            return np.array([1,2,3,0])

    def retrieve_by_index(self,iter,ind):
        for i in ind:
            yield iter[i]

    def _horz_sum(self,y_val=0,gr_total=0,pt_total=0):
        try:
            # min_max = np.array([int((y_val-ln[1])/ln[0]) if int((y_val-ln[1])/ln[0]) <= self.img.shape[1]-1 else self.img.shape[1]-1 for ln in self.v_lines])
            min_max = np.ones(2)
            for i,ln in enumerate(self.v_lines):
                x = int((y_val-ln[1])/ln[0])
    
                if x >= self.img.shape[1]-1:
                    x = self.img.shape[1]-1
                elif x < 0:
                    x = 0
                min_max[i] = x
        except OverflowError:
            min_max = np.array([self.v_lines[0][1],self.v_lines[1][1]])
        min_max = np.sort(min_max)
        if min_max[0] >= self.img.shape[1]-1 or min_max[1] <= 1:
            ic(gr_total,pt_total)
            return gr_total,pt_total
        row_pts = abs(self.lr_mat[y_val, int(min_max[0])] - self.lr_mat[y_val, int(min_max[1])])
        tot_pts = np.diff(min_max)[0]
        if y_val == self.lr_mat.shape[0]-1:
            return gr_total+row_pts,pt_total+tot_pts
        # cv2.line(self.disp_im,(int(min_max[0]),y_val),(int(min_max[1]),y_val),(0,255,255),2)
        return self._horz_sum(y_val+1,gr_total+row_pts,pt_total+tot_pts)

    def _calc_row_disparity(self,pt1,pt2):        
        td_disparity = abs(vpc.twoD_to_3D(pt1) - vpc.twoD_to_3D(pt2)).ravel()
        return td_disparity[0] 

    def test(self):
        self.find_rows()
        

def test(img):
    # ic.disable()
    # img_interact = image_interaction(data['imgs'][0])
    # img_interact.prep_plot()
    # test_annots = [[[ 93.69354839 ,350.59677419],
    # [172.40322581,   2.20967742],
    # [190.46774194,   4.79032258],
    # [123.37096774, 357.0483871 ]],
    # [[140.14516129, 346.72580645],
    # [204.66129032,   2.20967742],
    # [222.72580645,   2.20967742],
    # [187.88709677, 354.46774194]],
    # [[198.20967742, 349.30645161],
    # [249.8225806, 2.20967742],
    # [278.20967742,6.08064516],
    # [239.5, 357.0483871 ]],
    # [[261.43548387, 351.88709677],
    # [288.53225806, 0.91935484],
    # [307.88709677, 0.91935484],
    # [310.46774194, 357.0483871]]]

    test_annots = [[[ 2.08064516, 47.37096774], [30.46774194, 0.91935484], [39.5, 0.91935484], [3.37096774, 82.20967742]],
    [[2.08064516, 117.0483871], [56.27419355, 2.20967742], [69.17741935, 3.5], [4.66129032, 186.72580645]],
    [[0.79032258, 283.5], [93.69354839, 3.5], [106.59677419, 2.20967742], [3.37096774, 350.59677419]],
    [[29.17741935, 344.14516129], [127.24193548, 3.5], [144.01612903, 3.5], [66.59677419, 353.17741935]],
    [[97.56451613, 350.59677419], [169.82258065, 3.5], [191.75806452, 6.08064516], [123.37096774, 354.46774194]],
    [[146.59677419, 345.43548387], [207.24193548, 6.08064516], [221.43548387, 6.08064516], [181.43548387, 351.88709677]],
    [[211.11290323, 348.01612903], [254.98387097, -0.37096774], [274.33870968, 3.5], [247.24193548, 353.17741935]],
    [[269.17741935, 355.75806452], [288.53225806, 3.5], [309.17741935, 3.5], [298.85483871, 354.46774194]],
    [[327.24193548, 350.59677419], [329.82258065, 2.20967742], [346.59677419, 2.20967742], [360.79032258, 353.17741935]],
    [[384.01612903, 348.01612903], [364.66129032, 8.66129032], [386.59677419, 8.66129032], [413.69354839, 355.75806452]],
    [[440.79032258, 351.88709677], [404.66129032, 0.91935484], [429.17741935, 0.91935484], [469.17741935, 357.0483871]],
    [[496.27419355, 351.88709677], [442.08064516, 7.37096774], [466.59677419, 6.08064516], [524.66129032, 357.0483871]],
    [[549.17741935, 350.59677419], [483.37096774, 6.08064516], [502.72580645, 6.08064516], [580.14516129, 357.0483871]],
    [[611.11290323, 350.59677419], [516.91935484, 2.20967742], [542.72580645, 2.20967742], [635.62903226, 340.27419355]],
    [[638.20967742, 238.33870968], [559.5, 3.5], [586.59677419, 2.20967742], [635.62903226, 158.33870968]],
    [[634.33870968, 106.72580645], [598.20967742, 3.5], [617.56451613, 6.08064516], [636.91935484, 46.08064516]]]
    # return proc_annotations(img,img_interact.annotations).json_info
    return proc_annotations(img,test_annots).json_info

def video_main(data=None):
    vids = prep('sample',"C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Images/Drone_files/Winter_Wheat_vids") if type(data) == type(None) else data 
    cap = vids['vids'][0]
    ret, frame = cap.read()
    main(frame)
    cap.release() 

def main(img):
    img_interact = image_interaction(img)
    img_interact.prep_plot()
    json_path = write_annotations(proc_annotations(img,img_interact.annotations).json_info)
    return json_path

def test_without_writing(img):
    img_interact = image_interaction(img)
    img_interact.prep_plot()
    return proc_annotations(img,img_interact.annotations).json_info

if __name__ == "__main__":
    from prep_input import interpetInput as prep 
    from display_frames import display_dec as disp

    drones = 'Test_imgs/winter_wheat_stg1'


    data = prep('sample',drones)
    # test_without_writing(data['imgs'][0])
    # main(data['imgs'][0])
    write_annotations(test(data['imgs'][0]))

    # read the first frame of the video
    # write_annotations(proc_annotations(frame,real_annotations(frame)).json_info)
    # write_annotations(test(frame))




    #### TESTS: #####
    # type_handle_test(img_interact)




    