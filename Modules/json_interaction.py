"""
This will govern and format all the json interations and data managment within this project
 Module abilities:
  - Generate a new json file that is in the correct format for accessing the data
  - Data accessing
  - Data writing 
  - Data updating


"""
# json_interaction

import json
import warnings
import numpy as np
from icecream import ic

class load_json(object):
    def __init__(self,json_path = "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/python_soybean_c/Modules/Camera_Calib/Cam_specs_dict.json",dict_data=False):
        self.path = json_path
        self.dict_data = dict_data
        self.load_data()
    # The json file should contain the following parameters:
    # - Focal length
    # - Field of view
    # - Aspect ratio
    # - Sensor resolution (pixels)
    # - Sensor resolution (mm)
    # - Mount height from neutral point (mm)
    # - Mount angle from neutral point degrees defualt
    def load_data(self):
        with open(self.path) as f:
            self.data= json.load(f)["data"]
            f.close()

    # # This function will find the index of the dictionary in the file that contains the key 
    # def find_key(self,key, ret_val = False):
    #     with open(self.path) as f:
    #         data = json.load(f)["data"]
    #         if self.dict_data:
    #             for k in data.keys():
    #                 if key in data[k].keys():
    #                     val = data[k][key]
    #                     f.close()
    #                     if ret_val:
    #                         return val
    #                     return k
    #         else:
    #             for i,db in enumerate(data):
    #                 if key in db.keys():
    #                     f.close()
    #                     if ret_val:
    #                         return db[key]
    #                     return i
            
    #         warnings.warn('The key was not found in the file', UserWarning)

    #         f.close()
    #     return -1

    def find_key(self,key, ret_val = False):
        return self._dict_search(key,ret_val=ret_val)

    # this is a bfs search that will find the key in the json file 
    def _dict_search(self,des_key,ret_val=False):
        with open(self.path) as f:
            data = json.load(f)["data"]
            # the fronteir will contain lists of keys that corespond to the location in the data
            if des_key in data.keys():
                if ret_val:
                    return data[des_key]
                return [des_key]

            frontier = list([key] for key in data.keys() if isinstance(data[key],dict))

            assert len(frontier) > 0, "The json file is not in the correct format"
            while True:
                keys = frontier.pop(0)
                val = data
                for key in keys:
                    val = val[key]
                if des_key in val.keys():
                    f.close()
                    if ret_val:
                        return val[des_key]
                    return keys+[des_key]
                elif isinstance(val,dict):
                    for n_key in val.keys():
                        if isinstance(val[n_key],dict):
                            frontier.append(keys + [n_key])
                if len(frontier) == 0:
                    warnings.warn('The key was not found in the file', UserWarning)
                    f.close()
                    return -1
                    

            
    def update(self,d,v, delete):
        assert isinstance(d,dict), "The input must be a dictionary"
        assert len(v) > 1, "The input must be a list of length greater than 1"

        if len(v) == 2:
            if delete:
                d.pop(v[0],None) 
            else:
                d.update({v[0]:v[1]})

            return d
        d.update({v[0]:self.update(d[v[0]],v[1:],delete=delete)})
        return d

    def update_duplicate(self,data,proposal,delete=False):

        ic("updating duplicates")
        ret = {}
        for key, val in proposal.items():
            location = self._dict_search(key)
            if location != -1:
                location.append(val)
                data = self.update(data,location, delete=delete)

                # data[ind][key] = val
            else: 
                ret[key] = val
        if delete and ret != {}:
            warnings.warn("The key was not found in the file")
        return (ret,data)

    def delete_key(self,key):
        with open(self.path,'r+') as f:
            ic("Updating json file")
            cur_data = json.load(f)
            if type(key) == list:
                for k in key:
                    self.update_duplicate(cur_data["data"], {k:None}, delete=True)
            else:
                self.update_duplicate(cur_data["data"], {key:None}, delete=True)

            f.seek(0)

            json.dump(cur_data,f, indent=4)
            f.truncate()
            
            f.close()
        

        
    def write_json(self,cont_dict,data_title=None):
        for key in cont_dict.keys():
            if isinstance(cont_dict[key], np.ndarray): 
                ic("Changing to list")
                cont_dict[key] = cont_dict[key].tolist()
            if type(cont_dict[key]) == np.float64 or type(cont_dict[key]) == np.int64 or type(cont_dict[key]) == np.int32 or type(cont_dict[key]) == np.float32:
                cont_dict[key] = float(cont_dict[key])
        
        # ic(cont_dict)
        with open(self.path,'r+') as f:
            ic("Updating json file")
            cur_data = json.load(f)

            new_data, updated_data = self.update_duplicate(cur_data["data"],cont_dict)

            cur_data["data"] = updated_data

            assert isinstance(cur_data,dict) and isinstance(new_data,dict), "The input must be a dictionary"


            if len(new_data) > 0:
                print('The following keys were not found in the file and will be added: ',new_data.keys()) 
                if self.dict_data:
                    if data_title != None:
                        # if data_title in cur_data["data"].keys():
                        #     new_data = dict(cur_data["data"][data_title], **new_data)
                        if (val := self._dict_search(data_title)) != -1:
                            # val.pop()
                            val.append(new_data)
                            new_data = self.update(cur_data["data"],val,delete=False)
                        # cur_data["data"][data_title] = new_data
                    else:
                        warnings.warn("No data title was specified thus the data will be added to the root of the json file", UserWarning)
                        for key in new_data.keys():
                            cur_data["data"][key] = new_data[key]
                else:
                    cur_data["data"].append(new_data)

            f.seek(0)

            json.dump(cur_data,f, indent=4)
            f.truncate()
            
            # try: 
                # json.dump(cur_data,f, indent=4)
            # except TypeError:
            #     print(cur_data)
            #     print('The input is not a dictionary')
            f.close()



def main():
    loader = load_json("Adaptive_params/Adaptive_values.json",dict_data=True)
    # test the reader:
    val = loader.find_key("Focal length",ret_val=True)

    # test the writer:
    cont_dict = {"vanishing":np.array([1,2,3])}
    loader.write_json(cont_dict)

def update_test():
    loader = load_json("Adaptive_params/Adaptive_values.json",dict_data=True)

    # test the writer:
    loader.write_json({"test_val": {"v1":None, "v2":None}})
    loader.write_json({"v311":{"v3111":1}}, data_title="v1")
    loader.write_json({"v32":34})
    loader.write_json({"test_2": "test_2"})

    ic(loader._dict_search("v3111",ret_val=True))
    ic(loader._dict_search("v32",ret_val=False))
    ic(loader._dict_search("test_2",ret_val=True))

    loader.delete_key(["test_2", "v3111", "v32", "test_val"])

    


if __name__ == "__main__":
    # import sys,os
    # sys.path.append(os.path.abspath(os.path.join('.')))
    # import system_operations as sys_op
    # sys_op.system_reset()
    # main()
    update_test()