"""
Author: Joel Weber
Date: 01/07/2023
Description: This is the autonamic function of the system that should ensure seamless integration and running of the program. Much of the error handing and recalibration is performed
here
Quantum computing is cool

"""
import sys,os
import numpy as np
from icecream import ic

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
            "parameter path": ("Adaptive_params/Adaptive_values.json", "Current Settings")
        }
        for key, val in org_vals.items():
            self.params.update(key, val[0], val[1], universal=True)

def system_reset():
    system_reconfig().reset()


##### Custon Types #####
class type_handler(object):
    def __doc__(self):
        return "This module is the primary type checking and error handing site. This is built to make the development easier and centralize debugging"
    def __init__(self):
        self.type_dict = {
            "int": [int, np.int32, np.int64],
            "str": [str],
            "dictionary": [dict],
            "iterable": [tuple,list,np.ndarray]

        }
    def _isIterable(self,value):
        # ic(type(value))
        return type(value) in self.type_dict["iterable"]
    def shape(self,value):
        if self._isIterable(value):
            if type(value) == dict:
                return len(value.keys())    
            else: 
                return self._find_not_iterable(value)
    # Finds the lowest no iterable value in the iterable through recursion. Each return value yeilds the length of the current iterable and the length of it
    def _find_not_iterable(self,value):
        if not self._isIterable(value):
            return None
        elif type(value) == dict:
            pass
        else:
            # shape = filter(lambda i,v: v != None ,map(self.find_not_iterable,value))
            # Maps the shape of the iterable into (index within parent iterable, (iterable length, iterable))
            shape = [(i, self._find_not_iterable(v)) for i,v in enumerate(value) if self._isIterable(v)]
            if len (shape) == 0:
                return [len(value)]
            return [len(value),shape]
        
        

class generic_type(type_handler):
    def __doc__(self):
        return "This is the generic type that is used as the base for all other types. This is used to ensure that all types have the same basic functionality"
    def __init__(self):
        self.type = None
        self.similar_types = list()
        self.value = None
        super().init()
    def __repr__(self):
        return "generic_type()"
    def __len__(self):
        return len(self.value)


class active_type_checks(type_handler):
    def __doc__(self):
        return "Active checks that are always applied to ensure that the method yeilds a reasonable value"
    def __init__(self):
        pass

class implemenation_type_tests(type_handler):
    def __doc__(self):
        return "This holds the different verifications that the all type checking is passing successfully. There are two types of tests: 1. Implementation tests with pontially problematic types values and "
    def __init__(self):
        self.tests = {
           "iterable": self.iterable_test,
           "shape": self.shape_test
        }
        super().__init__() 
    def iterable_test(self):
        pass 
    def shape_test(self):
        tests = [
            [0],
            [1,2,3,4,5],
            [[1,2,3],[4,5,6],[7,8,9]],
            [[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]],
            [[1,2,3],[1,2,3,4,5],[1,2],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]],
            [1,2,3,[1,2,3,[1,2,3]]],
            [[1,2,3],[1,2,3,4,5],[1,2],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[1,2,3,[1,2,3,[1,2,3]]]]
        ]
        for i,test in enumerate(tests):
            try:
                ic(super().shape(test))
            except Exception as e:
                raise Exception("Shape test failed on test {} with error: {}".format(i,e))

    def test_all(self):
        for test in self.tests:
            test()
        
    

if __name__ == "__main__":
    system_reset()
    
## Implementation tests ## 
    # implemenation_type_tests().shape_test()
