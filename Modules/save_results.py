"""
Author: Joel Weber
Date: 12/06/2023
Description: Take the output on of the processed image function and save the results to a desitnation filt
"""
import cv2
import sys,os
# from PIL import Image
from datetime import datetime
from icecream import ic
import re
import numpy as np
import shutil 

"""
This will determine the correct action for the input type that it provided
These are the following types of data input that can be processed:
    - Single image 
    - List of images 
    - Single video 
    - Multple videos

*Note all data will be saved to the desination folder by the following rules
    saved as the data_title or date_type_save# if no title is provided
    if there are multiple images or videos they are archived into a folder with the determined title and labeled consecutively as result_1 -> result_len(data)
    by default all files are saved as .jpg for images and .mp4 for videos

*Note all data must be in an image form that is comprehensible by the cv2 library there is no support for path saving since the usecase is not essential for this project
*Note path to the desination folder should be absolute and not relative to the current working directory
"""
class data_save(object):
    def __init__(self):
        self.actions = {
            "img": self.save_img,
            "imgs": self.save_imgs,
            "vid": self.save_vid,
            "vids": self.save_vids
        }
        self.params_save = save_params()


    def check_overwrite(self, path):
        if self.verify_success(path, True):
            print("File already exists. Would you like to overwrite it? (y/n)") 
            if input() != "y":
                print("File not saved please enter a uniqie file name")
                return True
            return False

    def save_data(self, data, dtype, des_path = "", data_title = None, save_type = None):
        self.data_title = data_title if data_title else "test_{}".format(self.get_time())
        self.des_path = os.path.abspath(des_path)

        self.params_save.make_folder(os.path.abspath(des_path), data_title)

        self.des_path = self.params_save.folder_path
        self.save_type = save_type
        self.dtype = dtype
        self.data = data
        self.actions[self.dtype]()

    def save_img(self):
        path = self.gen_file_name("result_{}".format(self.data_title))
        if self.check_overwrite(path):
            return

        cv2.imwrite(path,self.data)
        self.verify_success(path)

    def save_imgs(self):
        #  create directory for images
        # self.des_path = self.gen_file_name(folder=True)

        for i,img in enumerate(self.data):
            path = self.gen_file_name("result_{}".format(i))
            cv2.imwrite(path,img)
            assert self.verify_success(path), "image was not saved correctly"
    
    # The video will be a list of frames and we must convert it into a mp4 file
    def save_vid(self):
        path = self.gen_file_name("result_{}".format(self.data_title))
        self.write_vid(path,self.data)
    
    def save_vids(self):
        # self.des_path = self.gen_file_name(folder=True)
        for i,vid in enumerate(self.data):
            self.write_vid(self.gen_file_name("result_{}".format(i)),vid)

    def write_vid(self,path,vid,fps = 30.0):
        # Set up video writer
        cap = None
        
        if type(vid) == type([]) or type(vid) == type(np.array([])) or type(vid) == tuple:
            frame_height, frame_width = vid[0].shape[:2]
            ic(frame_height, frame_width, vid[0].shape)

        else:
            ic(cap)
            cap = vid

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Change codec to the desired one
        out = cv2.VideoWriter(path, fourcc, fps, (frame_width, frame_height))
        
        # Loop through frames and write them to the output video
        if cap != None:
            fr =0 
            while cap.isOpened():
                print(fr)
                fr+=1
                
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break

                # cv2.imshow('frame', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            
            # Release video capture and writer, and close windows
            cap.release()
            out.release()
            # cv2.destroyAllWindows()
        else:
            for frame in vid:
                out.write(frame)
            out.release()

        assert self.verify_success(path), "video was not saved correctly"
        


    # Use the current time+test if there is no data_title
    def gen_file_name(self, data_title = None):
            
        if data_title == None:
            data_title = self.data_title

        if self.save_type == None:
            if "vid" in self.dtype:
                save_type = ".mp4"
            else:
                save_type = ".jpg"

        # if folder:
        #     path = os.path.join(self.des_path, data_title)
        #     print(path)
        #     # make folder in the desination path
        #     if not os.path.exists(path):
        #         os.mkdir(path)
        #     else: 
        #         raise Exception("Folder {} already exists please enter a uique folder name".format(path))

        #     assert self.verify_success(path), "Folder was not created correctly"
        #     return path

        return os.path.join(self.des_path, data_title+save_type)

    # ** This mode of saving may need to change if this is ever run in an integrated system
    def get_time(self):
        time = datetime.now()
        # repalce all non alphanumeric characters with _
        time = time.strftime("%Y-%m-%d %H:%M:%S")
        return re.sub(r'\W+', '_', str(time))


    def verify_success(self,path, test=False):
        if os.path.exists(path):
            if not test:
                print("File saved successfully at {}".format(path))
            return True
        # raise Exception("File was not saved correctly")
        return False

class save_params(object):
    def __doc__(self):
        return "Utility: Save the parameters that are ascociated with the data that is being saved. A folder for the test is create along with a copy of the parameters json file."
    
    def __init__(self, json_path = "Adaptive_params/Adaptive_values.json"):
        self.json_path = os.path.abspath(json_path)
        self.folder_path = None

    def make_folder(self ,des_path, data_title):

        path = os.path.join(des_path, data_title)
        print(path)
        # make folder in the desination path
        if not os.path.exists(path):
            os.mkdir(path)
        else: 
            raise Exception("Folder {} already exists please enter a uique folder name".format(path))

        assert os.path.exists(path), "Folder was not created correctly"
        self.folder_path = path
        self._add_json(self.json_path, data_title)
   
    # Require path to the current paramter json file being used 
    def _add_json(self, json_path, data_title):
        # copy the json file to the folder 
        new_path = os.path.join(self.folder_path, data_title+".json")
        shutil.copy(self.json_path, new_path)
        assert os.path.exists(new_path), "Json file was not copied correctly"


    
        

    
#### Tests ####
def test_img(imgs, saver):
    img = cv2.line(imgs[0],(0,0),(511,511),(255,0,0),5)
    saver.save_data(img,"img", "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/python_soybean_c/saved_tests", "test_img")

def test_imgs(imgs, saver):
    saver.save_data(imgs,"imgs", "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/python_soybean_c/saved_tests", "tested_imgs") 

def test_vid(vids, saver):
    saver.save_data(vids,"vid", "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/python_soybean_c/saved_tests", "test_vid")

def test_vids():
    pass
    
if __name__ == "__main__":
    import system_operations as sys_op
    sys_op.system_reset()
    from prep_input import interpetInput as prep 
    from display_frames import display_dec as disp
    from Image_Processor import apply_funcs as pre_process 

    drones = 'Test_imgs/winter_wheat_stg1'
    # This is an abosolute path since the video files are too large to be uploaded to github
    vids = "C:/Users/joelw/OneDrive/Documents/GitHub/Crop-row-recognition/Images/Drone_files/Winter_Wheat_vids"

    # imgs = prep('sample',drones)["imgs"]    
    vids = prep('sample',vids)["vids"]

    im_saver = data_save()

    # test_img(imgs, im_saver)
    # test_imgs(imgs, im_saver)
    test_vid(vids[0], im_saver)
    # print(vids)


    
    # print(dataCont)


# joel is a good programer XD :)    