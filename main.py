
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
    # local_reqs = {"prep", "disp", "proc", "ap", "noise", "load_json"}
    # This is where all the classes or functions of a module will be located. It will point to its parent and will simply need to be named as a global variable 
    # reqs = pip_reqs.union(local_reqs)

    try:
        for req in pip_reqs:            
            globals()[req] = modules[req]
    except KeyError:
        warnings.warn("The module {} is not present".format(req))


        
class apply_funcs(object):
    def __init__(self):
        self.funcs = {
            "noise": {}
        }

    def gen_funcs(self, des = None):
        pass

def main():
    # Call the primary detector module
    from Algorithims_tst.prim_detector import main as mn
    # Run the module which will begin the process of the entire program
    mn()




if __name__ == "__main__":

    main()