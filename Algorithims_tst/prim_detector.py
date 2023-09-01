"""
Author: Joel Weber
Title: Prim_detector.py
Overview: This is the primary control module that serves as the initail delegation hub for all the program functions. From here the test data is loaded and passed to the appropriate modules.
Issues: This is definitely an unfinished module and may be prone to unexpectedly crash however work is in progress to establish the detector ideas and then to preform rigourous bug elimaination at a 
later time when the project is more mature. 
"""
# edge_detection
# This is the research program the will test the viability of edge detection upon multipe hieristics
# There will also be some testing for the optimal image preprocessing

import cv2
import numpy as np
import path
import sys,os
from icecream import ic
import time
import warnings


try:
    from inceremental_masks import hough_assessment as hough    
except:
    from Algorithims_tst.inceremental_masks import hough_assessment as hough 
# import contour_detect as cont

sys.path.append(os.path.abspath(os.path.join('.')))

from Modules.prep_input import interpetInput as prep 
from Modules.display_frames import display_dec as disp
from Modules.Image_Processor import apply_funcs as pre_process 

import system_operations as sys_ops
from Algorithims_tst.aggregate_lines import ag_lines as agl
from Modules.strong_horz_kernel import kerneling as kern

from Adaptive_params import Adaptive_parameters as ap
from Algorithims_tst.MAKE_SOME_NOISE import noise 
try:
    # from consider_prev_lines import process_prev_lines as prev_lns
    from new_prev_lines import process_prev_lines as prev_lns
except:
    # from Algorithims_tst.consider_prev_lines import process_prev_lines as prev_lns
    from Algorithims_tst.new_prev_lines import process_prev_lines as prev_lns

params = ap.param_manager()
prev = prev_lns(avg=True)

def load_modules(module_list):
    reqs = {"cv2", "np", "time", "ic", }


def edge_detection(src_im, des = ["sobelx","sobely","sobelxy","laplacian"]):

    #Sobel Edge Detection:
    def sobel():
        ret = {}
        # img = pre_process(src_im, des=["blur", "gray"])
        img = pre_process(src_im, des=["kernel"])
        def testKernel():
            ret = {"img":src_im,"Radical":cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=31)}
            for i in range(13, 21, 2):
                ret["sobelx{}".format(i)] = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=i)
            return ret

        if "sobelx" in des:
            ret["sobelx"] = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=15)
        if "sobely" in des:
            ret["sobely"] = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=15)
        if "sobelxy" in des:
            ret["sobelxy"] = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=15)

        return ret

        # return (sobelx,sobely,sobelxy)
    # Laplacian Edge Detection:
    def laplacian():
        img = pre_process(src_im, des=["kernel", "gray"])
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        return laplacian
    sobel_im= sobel()

    ret = sobel()
    # return {'sobelX':sobel_im[0],'laplacian':laplacian()}
    return{'canny':pre_process(src_im,des=['kernel','mask']),'org':pre_process(src_im,des=['kernel'])}



def save_lns(data_name = 'winter_wheat.csv'):
    from Algorithims_tst.data_base_manager import data_base_manager as dbm
    return dbm(path = "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/python_soybean_c/saved_tests/saved_lines", data_name = data_name)

def save_sample(data,**kwargs):
    s_type = kwargs.get("s_type","vid")
    s_data = data[list(data.keys())[0]]
    from Modules.save_results import data_save as ds
    saver = ds()
    ic(type(s_data), len(data), data.keys())
    if s_type == "vid":
        saver.save_data(s_data,"vid", "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/python_soybean_c/saved_tests", "detected_vid_untrammeld")
        saver.save_data(s_data,"imgs", "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/python_soybean_c/saved_tests", "detected_imgs_untrammeld")
    elif s_type == "img":
        saver.save_data(s_data,"img", "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/python_soybean_c/saved_tests", "detected_img")


"""
The filters that the image passes through 
 1. Resize to (640,360)
 2. Mask 
 3. Blur 
 4. 
"""
def inc_color_mask(img, **kwargs):
    ic(params.access("row_num"))
    # ic.disable()
    # make defaults for the kwargs:
    upper = kwargs.get("upper",[75,255,255]) 
    lower = kwargs.get("lower",[30,50,50]) 
    resolution = kwargs.get("resolution",10)
    noise_tst = kwargs.get("noise_tst",0)
    hough_cmd = kwargs.get("hough_cmd","avg")
    # Disp_cmd values that get a result: "final" (displays on the final frame) "disp_steps" (displays the steps of the process) "none" (displays nothing)
    disp_cmd = kwargs.get("disp_cmd",None)
    # The database object should be passed in as a parameter
    save_lns = kwargs.get("save_lns",None)
    # Detection is a video
    video = kwargs.get("video",False)
    # prev_lines = kwargs.get("prev_lines",None)
    row_num = params.access("row_num")


    no_noise_org = pre_process(img.copy(), des=["resize"])
    if noise_tst != 0:
        img = noise(img, noise_pts=noise_tst)
    no_noise = pre_process(no_noise_org.copy(), des=["mask"])
    img = pre_process(img, des=["resize"])

    # Initailize the contour class:

    def map():
        iterations = np.array([(upper[0]-lower[0])/resolution,(upper[1]-lower[1])/resolution,(upper[2]-lower[2])/resolution])
        return iterations
    increments = map()
    org = pre_process(img, des=["kernel","mask"],upper=upper,lower=lower,colors=True)
    ret = {"org":img ,"mask_init":org}
    # ret["org"] = img  
    # ret["mask_init"] = org

    # To group lines change the command from "none" to "avg"
    # add "disp_steps" to display the layers of masks to the output
    imgs = hough(img,org,no_noise,hough_cmd, disp_cmd)
    # imgs = hough(img,org,no_noise,hough_cmd,"disp_steps")
    up = upper[0]
    low= lower[0]


    def get_masks(org,up,low):
        cur_msk = pre_process(org, des=["blur","mask","kernel"],upper=up,lower=low,colors=True)
        bin_mask = pre_process(org, des=["mask","kernel"],upper=up,lower=low)
        return bin_mask,cur_msk

    total_mask = get_masks(org,upper,lower)
    imgs.slope_filter(total_mask[0],total_mask[1])

    num_splits = params.access("incremental_masks")
    color_range = params.access("color_range") 
    # for i in range(1,resolution+1):
    for v in imgs.color_splits(num_splits,color_range):

        up = np.array([v[1],upper[1],upper[2]])
        low = np.array([v[0],lower[1],lower[2]])
        bin_mask,cur_msk = get_masks(org,up,low) 

        # Run the blob detections on the image
        # Initalize the contour class:
        # Group the lines that are deemed good by slope and  intercept

        imgs.slope_filter(bin_mask,cur_msk)
        ret["mask{} {}".format(v[0],v[1])] = cur_msk


    # Create a mask for the image
    if save_lns != None:
        ic(np.array(imgs.good_lns).shape)
        save_lns.update_csv(imgs.good_lns)
    
    if hough_cmd == "avg":
        res_img = no_noise_org.copy()
        if len(imgs.good_lns) > 1:

            final_rows = agl(img, imgs.good_lns)

            if video and type(prev.prev_lines) != type(None):
                # ic(prev.prev_lines)
                groups, score , best_lines = final_rows.calc_pts([0,img.shape[0]],True)
                if type(best_lines) == type(None):
                    gr_lines = prev.prev_lines
                else:
                    # top_inds = np.argsort(np.array(score))[::-1][:round(len(score)/3)]
                    num_rows_filt = np.where(np.array([len(i) for i in groups]) > row_num/2)[0]
                    ic(num_rows_filt)
                    if len(num_rows_filt) > 0:
                        best_grps = [groups[i] for i in num_rows_filt]
                    else: 
                        warnings.warn("No rows were detected in the image")
                        best_grps = [best_lines] 

                    gr_lines = prev.best_match(best_grps)[2]
                    prev.prev_lines = gr_lines

                final_frame = imgs.group_lns(gr_lines,res_img,grp = False, avg=False, prev=True)
            else:
                pts, gr_lines = final_rows.calc_pts([0,img.shape[0]],False)
                prev.prev_lines = gr_lines
                final_frame = imgs.group_lns(gr_lines,res_img,grp = False)

        elif type(prev.prev_lines) != type(None):
            disp_lines = [[np.nan, ln[1]] for ln in prev.prev_lines]
            final_frame = imgs.group_lns(disp_lines,res_img,grp = False, avg=False, prev=True)

        else: 
            final_frame = res_img
            
        imgs.ret["final"] = final_frame

        if disp_cmd == "final":
            return {"final":final_frame}

        # ic(gr_lines)


    return imgs.ret

def calibrate(data):
    import random
    from Modules.image_annotation import video_main as vm
    vm(random.choice(data["vids"]))

def frames_of_concern(dataCont, frame_range = (280, 307), vid_ind = 0, **kwargs):
    # verify the the provided slice is within the video range
    assert frame_range[0] <= frame_range[1], "The frame range is not valid"
    assert frame_range[0] >= 0 and frame_range[1] <= dataCont["vids"][vid_ind].get(cv2.CAP_PROP_FRAME_COUNT), "The frame range is not within the video range"
    cap = dataCont["vids"][vid_ind]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0]) 
    for i in range(frame_range[0], frame_range[1]):
        ret, frame = cap.read()
        dataCont["imgs"].append(frame)

    dataCont["vids"] = []
    disp(dataCont, inc_color_mask , **kwargs)

class sample_modes(object):
    def __init__(self, data=None, vid_path ="C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Images/Drone_files/Winter_Wheat_vids/vids/vid_0", img_path = 'Test_imgs/winter_wheat_stg1'):
        if data == None:
            vid_cont = prep('sample',vid_path)
            img_cont = prep('sample',img_path)
            self.data = {
                "vids":vid_cont,
                "imgs":img_cont
            }
        else: 
            self.data = {
                "vids":data,
                "imgs":data
            }
        assert type(self.data["vids"]) == dict and type(self.data["imgs"]) == dict, "The data must be a dictionary with the keys 'vids' and 'imgs'"
        self.kern = kern()
    # This will save the detected lines to a csv file where the data can be easily loaded for testing purposes
    def save_lines_csv(self, sample_type = "vids", csv_fname = "winter_wheat.csv"):
        data_base = save_lns(data_name=csv_fname)
        return disp(self.data[sample_type], inc_color_mask, hough_cmd = "avg", disp_cmd = "disp_steps", save_lns = data_base)
    
    def detection_on_video(self, **kwargs):
        # if "video" not in kwargs:
        #     kwargs["video"] = True
        return disp(self.data["vids"], inc_color_mask ,**kwargs )

    def detection_on_image(self, **kwargs):
        # if "video" not in kwargs:
        #     kwargs["video"] = False 
        return disp(self.data["imgs"], inc_color_mask , **kwargs)
        
    def calibrate(self, dataCont,**kwargs):
        return calibrate(dataCont)

    def specific_frames(self, d_key='vids', frame_range = (280, 307), vid_ind=0, **kwargs):
        return frames_of_concern(self.data[d_key], frame_range=frame_range, vid_ind=0, **kwargs)

    # this requires the row width and spacing to be provided in to caluclate what the kernel breadth should be 
    def kernel(self, d_key='vids', **kwargs):
        return disp(self.data[d_key], self.kern.apply_kernel, **kwargs)

    # def save_sample(self, )

    



def main(data_path=None,**kwargs):
    testing = sample_modes()
    # testing.detection_on_video()
    testing.specific_frames(video=True)
    # testing.save_lines_csv(csv_fname="corn.csv")

if __name__ == "__main__":
    import system_operations as sys_op
    sys_op.system_reset()
    main()