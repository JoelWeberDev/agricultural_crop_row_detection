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
        # ic("Loading camera parameter")
        with open(self.path) as f:
            self.data= json.load(f)["data"]
            f.close()

    # This function will find the index of the dictionary in the file that contains the key 
    def find_key(self,key, ret_val = False):
        with open(self.path) as f:
            data = json.load(f)["data"]
            if self.dict_data:
                for k in data.keys():
                    if key in data[k].keys():
                        val = data[k][key]
                        f.close()
                        if ret_val:
                            return val
                        return k
            else:
                for i,db in enumerate(data):
                    # ic(db)
                    if key in db.keys():
                        f.close()
                        if ret_val:
                            return db[key]
                        return i
            
            warnings.warn('The key was not found in the file', UserWarning)

            f.close()
        return -1

    def update_duplicate(self,data,proposal):
        ret = {}
        for key in proposal.keys():
            if (ind := self.find_key(key)) != -1:
                data[ind][key] = proposal[key]

            else: 
                ret[key] = proposal[key]
        return (ret,data)
        
    def write_json(self,cont_dict,data_title=None):
        for key in cont_dict.keys():
            if isinstance(cont_dict[key], np.ndarray): 
                ic("Changing to list")
                cont_dict[key] = cont_dict[key].tolist()
            if type(cont_dict[key]) == np.float64 or type(cont_dict[key]) == np.int64 or type(cont_dict[key]) == np.int32 or type(cont_dict[key]) == np.float32:
                cont_dict[key] = float(cont_dict[key])
        

        with open(self.path,'r+') as f:
            ic("Updating json file")
            cur_data = json.load(f)

            new_data, updated_data = self.update_duplicate(cur_data["data"],cont_dict)

            cur_data["data"] = updated_data
            # ic(updated_data,new_data)

            if len(new_data) > 0:
                print('The following keys were not found in the file and will be added: ',new_data.keys()) 
                if self.dict_data:
                    if data_title != None:
                        if data_title in cur_data["data"].keys():
                            # ic(cur_data["data"][data_title])
                            new_data = dict(cur_data["data"][data_title], **new_data)
                        cur_data["data"][data_title] = new_data
                    else:
                        raise ValueError("The data_title must be specified if the json file is a dictionary")
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

    def create_json(self):

        pass


def main():
    loader = load_json("Adaptive_params/Adaptive_values.json",dict_data=True)
    # test the reader:
    val = loader.find_key("Focal length",ret_val=True)
    ic(val)

    # test the writer:
    cont_dict = {"vanishing":np.array([1,2,3])}
    loader.write_json(cont_dict)


if __name__ == "__main__":
    main()