# resize_images
# Module Outline:
# - Take the input frame an outputs one that is sized to more processable dimentions
# - Can also preform a image crop to focus on only relevant areas
# 

import cv2
import numpy as np
import prep_input as prep

def resize_images(img, **kwargs):
    # Check for arguments
    def checkArgs():
        for key, arg in kwargs.items():
            if key == 'size':
                try:
                    size = arg
                    if len(size) != 2:
                        raise Exception('Invalid size')
                except:
                    print('Invalid size')
            elif key == 'crop':
                try:
                    crop = arg
                    if len(crop) != 4:
                        raise Exception('Invalid crop')
                except:
                    print('Invalid crop')
            else:
                print('Invalid argument')
    
    # Resize image
    def resize(img, size):
        return cv2.resize(img, size)
    
    # Crop image
    def crop(img, crop):
        return img[crop[0]:crop[1], crop[2]:crop[3]]
    
    # Check for arguments
    checkArgs()
    
    # Resize image
    if 'size' in kwargs:
        img = resize(img, kwargs['size'])
    # Crop image
    if 'crop' in kwargs:
        img = crop(img, kwargs['crop'])
    
    return img 
    
# Test
if __name__ == '__main__':
    import system_operations as sys_op
    sys_op.system_reset()
    # Get images
    images = prep.interpetInput(dataType='sample', path = 'Resized_Imgs')
    # Resize images
    for img in images['imgs']:
        print(img.shape)
        cv2.imshow('test', resize_images(img, size=(750,1000)))
        print(img.shape)
        cv2.waitKey(0)
        cv2.destroyAllWindows()