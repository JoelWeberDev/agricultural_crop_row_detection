"""
Author: Joel Weber
Title: Modules Import
Overview: This is the imports of all the modules that will be used in the project
"""
# modules_imp
class mm(object):
    def __init__(self):
        self._global_imports()
        # self._local_imports()
        # self.all_imps = {**self._global_vars, **self._local_vars}
        self.all_imps  = self._global_vars
        self.set_vars()

    def _global_imports(self):
    # Global imports
        import sys, os, cv2, math, random, numpy as np, time, json, matplotlib.pyplot as plt, warnings
        from icecream import ic
        self._global_vars ={
            "cv2": cv2,
            "np": np,
            "sys": sys,
            "os": os,
            "math": math,
            "random": random,
            "time": time,
            "json": json,
            "plt": plt,
            "warnings": warnings,
            "ic": ic
        }

    def _local_imports(self):
        # Local imports
        from Modules.prep_input import interpetInput as prep 
        from Modules.display_frames import display_dec as disp
        from Modules.Image_Processor import procImgs as proc
        from Modules.json_interaction import load_json

        from Adaptive_params import Adaptive_parameters as ap

        from Algorithims_tst.Group_lines import group_lns as grp
        from Algorithims_tst.inceremental_masks import hough_assessment as hough
        from Algorithims_tst.MAKE_SOME_NOISE import noise 
        from Algorithims_tst.prim_detector import pre_process as e_pre, edge_detection as egde
        


        self._local_vars = {
            "grp": grp,
            "hough": hough,
            "e_pre": e_pre,
            "edge": egde,
            "prep": prep,
            "disp": disp,
            "proc": proc,
            "ap": ap,
            "noise": noise,
            "load_json": load_json
        }

    def set_vars(self, set_global=True, set_local=True):
        # Locals:
        # if set_local:
        #     for key, val in self._local_vars.items():

        #         globals()[key] = val
        
        if set_global:
            for key, val in self._global_vars.items():
                globals()[key] = val

    def mods_to_json(self):
        from Modules.json_interaction import load_json
        self.mod_json = load_json("Modules/modules.json", dict_data=True)
        for key, val in self.all_imps.items():
            self.mod_json.write_json({key:str(val)}, data_title="Modules")
            # print(type(val))

    def add_to_globals(self):
        import re
        mod_imp.mods_to_json()
        mods = mod_imp.mod_json.data["Modules"]
        for key, val in mods.items():
            # print(val.strip("<class '").strip("'>"))
            # print(val)
            globals()[key] = val
            # literal  = ast.literal_eval(val)
            # print(literal)


if __name__ == "__main__":
    mod_imp = mm()
    # print(globals())
    # mod_imp.mods_to_json()
    mod_imp.add_to_globals()
else:
    import inspect
    caller_globals = dict(inspect.getmembers(inspect.stack()[1][0]))["f_globals"]
    print(caller_globals)


