"""
Author: Joel Weber
Date: 01/07/2023
Description: This is the autonamic function of the system that should ensure seamless integration and running of the program. Much of the error handing and recalibration is performed
here
Quantum computing is cool

"""
import sys,os
from Adaptive_params.Adaptive_parameters import param_manager as ap


class system_reconfig(object):
    def __doc__(self):
        return "Here we return the system to the original state that it was in before the program was run. This should reset the program in the event of a crash or premature exit."
    
    def __init__(self):
        self.params = ap()

    def reset(self):
        # Reset the parameters
        self._reset_json()
        print("System reset complete")
        # Reset the system
        # sys.exit()

    """
    The following paramters will be return to their original state regardless of what they were set in the intermediate process:
    1. alternative_path (str) -> 'Adaptive_params/Adaptive_values.json' , data_title = "user_input"
    """
    def _reset_json(self):
        # Reset the JSON file
        org_vals = {
            "parameter path": ("Adaptive_params/Adaptive_values.json", "user_input")
        }
        for key, val in org_vals.items():
            self.params.update(key, val[0], val[1], universal=True)

def system_reset():
    system_reconfig().reset()

if __name__ == "__main__":
    system_reset()