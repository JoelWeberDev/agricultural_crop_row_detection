# Module that imports images or video and converts that into a format condusive to opencv operation

# Module Predicate:
#   - Puts images or videos in the format readable by opencv
#       - Function Decomp:
#        - Test images: Load them from a folder and place them in an iterable format
#        - Live video: Segments the video into frames that will be seuqentially processed
#        - CV Format: Convert the input image into a fromat that can be processed by opencv
#   - Returns the input image or video in a format that can be processed by opencv

# Next Steps:
# - Implement funtion that converts frames into cv format and conversely generates other frames needed for the processing
# - Integrate with the main program
  # - Transform the dispaly functionality to work with the image format
  # - Import the module in the main program
# - Test with live video
# - Live capture feature enable

# How to use:
#   - Pass the folder path of your desired images to the interpet input function
#   - For a folder of samples pass the "sample" key word to the function.
#   Return values: 
    # The function will return a list of images that then can be passed to the display module;


import cv2
import numpy as np
import os

## LOAD DATA FROM SOURCE

# Determine image source type
def interpetInput(dataType='live', path =""):
    if dataType == 'sample':
        return loadSampleImages(path = path)
    elif dataType == 'live':
        return loadSampleImages(data= prepVideo())
    else:
        print('Invalid input source')

# Loading from sample folder
def loadSampleImages(**kwargs):

    imageMap = {'imgs': [], 'vids': [], 'other': []}

    # Check path integrity
    def checkArgs():
        for key, arg in kwargs.items():
            if key == 'path':
                try:
                    retrievePath(arg)
                except Exception as e:
                    print(e)
                    print('Invalid path')
            
            elif key == 'data':
                isLive(arg)
                

    def retrievePath(rel_path):
        print(rel_path)
        # Get absolute path
        abs_path = os.path.abspath(rel_path)
        # Get all files in the folder
        files = os.listdir(abs_path)
        for file in files:
            ap_path = abs_path + '/' + file
            if isImage(file):
                imageMap['imgs'].append(cv2.imread(ap_path))
            elif isVideo(file):
                imageMap['vids'].append(cv2.VideoCapture(ap_path))
            else:
                imageMap['other'].append(ap_path)

    # Create a dictionary to store the images and videos

    def isImage(file):
        if os.path.splitext(file)[1] in {'.jpg','.JPG','.jpeg','.jpe','.jp2','.tif', '.tiff','.sr','.ras','.pbm','.pgm','.ppm','.bpm','.png' }:
            return True
        return False

    def isVideo(file):
        # print(os.path.splitext(file)[1])
        if os.path.splitext(file)[1] in {'.avi', '.MP4', '.mov', '.mp4'}:
            return True
        return False
    
    def isLive(data):
        if data.isOpened():
            imageMap['vids'].append(data)

    # Takes a mix of files and sorts them into their respective categories

    checkArgs() 
    return imageMap

# Load video from live stream
def prepVideo():
    print('Loading video...')
    cap = cv2.VideoCapture(0)
    return cap

## TEST CASES ##
def mock_fxn_test(img):
    return cv2.resize(img, (0,0), fx=0.5, fy=0.5)

def test():
    import display_frames as disp
    target = 'Drone_images/Winter_Wheat_vids'
    # target = 'Resized_Imgs'
    datacont = interpetInput("sample", target)
    disp.display_dec(datacont, mock_fxn_test)
    # disp.display_dec(interpetInput('live'))

if __name__ == "__main__":
    import system_operations as sys_op
    sys_op.system_reset()
    test()