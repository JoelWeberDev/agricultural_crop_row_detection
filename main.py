
"""
Author: Joel Weber
Title: Main
Overview: This is the hub file that will control and deligate to all the other files in the project
Vision: 
 - We need to keep the system ordely and easy to follow so this should be where all the other files are called from
 - Test cases should be easy to switch between such that we can combine modules simply and efficently
 - All the returns for images should be similar and copatible.
"""
'''
Global tasks:
 - make the methods that should not be called from outside the module private
 - improve error handling such that many of the comatability issues will fix themselves
 - make generic data structures for images and groups of images that will be universal
'''

# from Modules.prep_input import interpetInput as prep
# from Modules.display_frames import display_dec as disp
# from 

import os
from icecream import ic
import shutil 

from Adaptive_params import Adaptive_parameters as ap
from Modules.prep_input import interpetInput as prep 
from Modules.display_frames import display_dec as disp
from Modules.Image_Processor import apply_funcs as pre_process 

"""
Class outline:
 - Load the images from the prep module
 - Display the images from the display module
 - Have a function that will load any function from module and combine it into one that will be
  passed to the display module
   - Group the modules by compatability
"""
def imp_mods(modules):
    import warnings
    pip_reqs = {"cv2", "np", "time", "ic" }
    local_reqs = {"prep", "disp", "proc", "ap", "noise", "load_json"}
    # This is where all the classes or functions of a module will be located. It will point to its parent and will simply need to be named as a global variable 
    reqs = pip_reqs.union(local_reqs)

    try:
        for req in pip_reqs:            
            globals()[req] = modules[req]
    except KeyError:
        warnings.warn("The module {} is not present".format(req))
    # ic(globals())


def main():
    # Import the modules
    # Call the primary detector module
    from Algorithims_tst.prim_detector import main as mn
    # Run the module which will begin the process of the entire program
    mn()


# sample data tests
from Modules.prep_input import interpetInput as prep
from Algorithims_tst import prim_detector as prim

# This is the primary testing platform for the system. Add the paths of images and videos to the test_paths list to test them. *Note if you have never done an annotation for them before you will need to do that first
class testing(object):
    def __init__(self):
        from Modules.save_results import data_save as save
        self.saver = save()
        self.test_paths = [
            # ["C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\corn","imgs"]
            ["C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\03-07-2023_transfer\\corn","vids"]
            # ["C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\Images\\Drone_files\\Winter_Wheat_vids","vids"]
            # ["Adaptive_params\\tests\\small_corn","imgs"]
            # ["Adaptive_params\\tests\\mid_corn","imgs"],
            # ["Adaptive_params\\tests\\small_soybeans","imgs"]
        ]    
        self.video_extensions = {".mp4", ".avi", ".mov",".MP4", ".AVI", ".MOV"}
        self.params = ap.param_manager()
        self.samples = prim.sample_modes()

        self.sample_funcs = {
            "image_detection": self.samples.detection_on_image,
            "video_detection": self.samples.detection_on_video,
            "specific_frames": self.samples.specific_frames,
            "calibrate": self.samples.calibrate
        }

    # *Note: For the data samples to work the images must be included in a folder labeled "imgs" and the json file must be labeled "test.json" videos must be labeled "vids"
    # Please pass all the arguments that you desire to be applied to the detection into this function as kwargs
    def image_tests(self, specific_frames=None, **kwargs):
        save = kwargs.get("save", False)
        save_lns = kwargs.get("save_lns", False)
        vids = []

        for test, data_type in self.test_paths:
            data_path = os.path.join(test, data_type)
            # ensure that the data path exists
            assert os.path.exists(data_path),f"The data path {data_path} does not exist"

            if data_type == "imgs":
                data_cont= prep("sample", data_path)
                self.samples.data["imgs"] = data_cont
                self.new_params(test, data_type, data_cont["imgs"])
                # samples = disp(data_cont, prim.inc_color_mask)
                self.apply_func("image_detection",**kwargs)

            elif data_type == "vids":
                ic(os.listdir(data_path))
                for i,sub_path in enumerate(os.listdir(data_path)):
                    if os.path.splitext(sub_path)[1] in self.video_extensions:
                        new_folder_path = os.path.join(data_path, f"vid_{i}")
                        os.mkdir(new_folder_path)
                        shutil.move(os.path.join(data_path, sub_path), new_folder_path)
                        ic(new_folder_path)
                        data_cont= prep("sample", new_folder_path)
                        des_path = new_folder_path
                    else: 
                        data_cont = prep("sample", os.path.join(data_path, sub_path))
                        des_path = os.path.join(data_path, sub_path) 

                    # Update the data to the samples object
                    self.samples.data["vids"] = data_cont
                    # vid_path = os.path.join(new_folder_path, sub_path)
                    # data_cont= prep("sample", vid_path)
                    self.new_params(des_path, data_type, data_cont["vids"])
                    if save_lns:
                        save_name = f"vid_{i}_lines.csv"
                        samples = self.save_lines(data_cont, save_name)

                    elif type(specific_frames) != type(None):
                        samples = self.apply_func("specific_frames",frame_range=specific_frames, vid_ind=i, **kwargs)

                    else:
                        samples = self.apply_func("video_detection",**kwargs)
                        # samples = disp(data_cont, prim.inc_color_mask, video = True, noise_tst = 0, disp_cmd = "final")

                    if save:
                        s_path = "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\python_soybean_c\\saved_tests"

                        assert type(samples) == dict, "The samples must be a dictionary" 
                        key = list(samples.keys())[0]
                        self.save_results(samples[key], s_path, title=f"{os.path.splitext(sub_path)[0]}_result", s_type = "vid")

                    vids.append(samples)

                    prim.prev.prev_lines = None

                # if save:
                #     s_path = "C:\\Users\\joelw\\OneDrive\\Documents\\GitHub\\Crop-row-recognition\\python_soybean_c\\saved_tests"
                #     self.save_results(vids, s_path, title=f"{sub_path[0]}_result", s_type = "vids")

                # self.new_params(test, data_type, data_cont["vids"])

            # samples = disp(imgs, inc_color_mask, disp_cmd = "final")

    def apply_func(self, func_name, **kwargs):
        assert func_name in self.sample_funcs.keys(), f"The function {func_name} is not a valid function"
        func = self.sample_funcs[func_name]
        return func(**kwargs)
     
    def new_params(self, fold_path, data_type, samples):
        import random
        from Modules import image_annotation as ia
        if data_type == "vids":
            pass
        # make a separate folder the stores a json file and the video
        json_file = self.contains_json(fold_path)
        if json_file != None:
            js_path = os.path.join(fold_path, json_file)
            # pm = param_manager(json_file)
        else:
            new_path = os.path.join(fold_path , "test.json")
            shutil.copy(self.params.path, new_path)
            json_path = ia.main(random.choice(samples),new_path) if data_type == "imgs" else ia.video_main(random.choice(samples),new_path)

            assert os.path.exists(new_path), "Json file was not copied correctly"
            js_path = new_path

        self.params.update("parameter path", js_path, title="Current Settings", universal=True)
        self._test_json_loads(js_path)

    def contains_json(self,path):
        for file in os.listdir(path):
            if file.endswith(".json"):
                return file 
        return None 

    def save_results(self,results, path, s_type = "imgs", title="row_detection_results" ):
        self.saver.save_data(results, s_type, path, title)

    def _test_json_loads(self, cur_json):
        param_path = os.path.abspath(os.path.realpath(self.params.path))
        desired_path = os.path.abspath(os.path.realpath(cur_json))
        alleged_path = os.path.abspath(os.path.realpath(self.params.access("parameter path",universal=True)))
        ic(param_path, desired_path, alleged_path)
        assert param_path == desired_path == alleged_path, "Current path json file and parameter path do not match" 

    def save_lines(self, data_cont, s_path='winter_wheat.csv'):
        save_lns = prim.save_lns(s_path)
        return disp(data_cont, prim.inc_color_mask, video = False, save_lns = save_lns , noise_tst = 0, disp_cmd = "final")
        

if __name__ == "__main__":
    import system_operations as sys_op
    sys_op.system_reset()
    td = testing()
    td.image_tests(noise_tst=0, specific_frames=(10,215), video = True, disp_cmd = "final")
    # td.image_tests(noise_tst=0, disp_cmd="final")
    # td.image_tests(noise_tst=0, disp_cmd="final", save=True)
    # td.image_tests(noise_tst=50000)
