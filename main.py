
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
class test_data(object):
    def __doc__(self):
        return "This is the primary testing hub of all the sample data that is currently tabulated and calibrated for the project."
    def __init__(self):
        from Adaptive_params import Adaptive_parameters as ap
        self.ap = ap
        self.ap_test = ap.testing()

        self.drones = 'Test_imgs/winter_wheat_stg1'
        self.vids = "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Images/Drone_files/Winter_Wheat_vids"

        self.test_paths = [
            "Adaptive_params\\tests\\small_corn",
            "Adaptive_params\\tests\\mid_corn",
            "Adaptive_params\\tests\\small_soybeans"
        ]    

    # The path should be linked to the data folder. If the folder has a JSON file that will be used as the calibration file otherwise the user will be reqired to preform a 
    # manual image calibration.
    def load_data(self, data_path):    
        from Modules.json_interaction import load_json
        import os

        imgs = os.listdir(data_path)

        json_path = self.ap_test.contains_json(data_path)
        # Load the images
        # Return the images and the calibration
        return imgs, json_path 

    def run_test(self, test_path):
        # Load the data
        imgs, cailb_json = self.load_data(test_path)

        # Run the main module
        from Algorithims_tst import prim_detector as pd

        self.ap.path = cailb_json

        pd.main(imgs)




if __name__ == "__main__":

    main()