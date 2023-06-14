import os, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

from moviepy.editor import VideoClip 
from moviepy.video.io.bindings import mplfig_to_npimage


# Module to display images and videos
# Function Outline:
  # - Recieve frame
  # - Allow for user interations
  # - Apply operations to frame

# How to use:
#  Desc: The display dec function takes data as an iterable and a function the will be applied to each frame of that data
#   the return of the function that you pass must be in the form of a dictionary or string indexed map with the name of the display
#   and the coresponding frame. The default false display org function can be passed as true if the original image should be displayed
#   as a reference for the processed one.
#   

# Display Video:
def display_dec(data, func = None,dispOrg = False,**kwargs):

    #  We need to create this to take any of the methods in which data will be supplied to us and turn it into a dictionary with titles
    def handle_dtype(data): 
        indexed_it = lambda d: {str(i):d[i] for i in range(len(d))}
        dat_fxns = {list: indexed_it, dict: lambda d: d, tuple: indexed_it,np.ndarray: indexed_it, "other": lambda d: {"only":d}}
        try:
            if type(data) in dat_fxns.keys():
                return dat_fxns[type(data)](data)
            else: 
                return dat_fxns["other"](data)
        except Exception as e:
            raise e("Invalid data type please enter {} is not valid".format(type(data)))

    def apply_func(sample):
        # Return a dicitionary that contais the image name and the corresponding image as the value
        if func:
            # display only the processed frame 
            
            if dispOrg:
                # display both the processed and original fram
                ret = handle_dtype(func(sample,**kwargs))
                ret["initial"] = sample
                return (func(sample,**kwargs)["initial"]) 
            return (handle_dtype(func(sample,**kwargs))) 
        return {"only":sample}
    # UI navigation: 
    # - 'q' to quit
    # - 'n' to go to next frame

    # Sample data structure: 
    def vid(sample):
        # anaylze the sample to determine how the data is stored
        while True:
            ret, img = sample.read()
            if ret:
                frames = apply_func(img)
                if b: break
                for title in frames.keys():       
                    cv2.imshow(title, frames[title])
                # cv2.waitKey(1)

            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return True
            elif cv2.waitKey(1) & 0xFF == ord('n'):
                break
        sample.release()
        cv2.destroyAllWindows()

    def img(sample):
        while True:
            cv2.imshow(list(sample.keys())[0], list(sample.values())[0]) 
            if cv2.waitKey(10) & 0xFF == ord('q'):
                return True
            elif cv2.waitKey(10) & 0xFF == ord('n'):
                return False

    def start_display(data):
        for key in data:
            # Sample will always be a singular image or video

            global b
            b = False
            for sample in data[key]:
                if key == 'imgs':
                    imgs = apply_func(sample)
                    # print(len(imgs))

                    if len(imgs) > 1:
                        dispMat(imgs)

                        if b: break
                        continue
                    if img(imgs): 
                        return
                    cv2.destroyAllWindows()

                elif key == 'vids':
                    if vid(sample): 
                        return
    start_display(data)

# Alternative display option: PyPlot 
# - Data Feed:
#     - Images - Numpy array format
#     - Videos - CV Cap format
b = False
def dispMat(imCol):
    sq = len(imCol)**(1/2)
    for i,imKey in enumerate(imCol.keys()):
        plt.subplot(round(sq),math.ceil(sq),i+1)
        try:
            rgbIm = cv2.cvtColor(imCol[imKey], cv2.COLOR_BGR2RGB)
            plt.plot(),plt.imshow(rgbIm) 
        except:
            rgbIm = imCol[imKey]
            plt.plot(),plt.imshow(rgbIm,cmap='gray') 
        # plt.figure(figsize=())
        plt.title(imKey)
        plt.connect('key_press_event', onPress)
        plt.connect('button_press_event', onClick)
    plt.show()


# UI Interface for matplotlib
def onPress(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'q':
        plt.close(event.canvas.figure)
        global b
        b = True
        raise Exception('User quit')
    elif event.key == 'n':
        plt.close(event.canvas.figure)
    elif event.key == 's':
        plt.savefig('test.png')
        print('saved')
        sys.stdout.flush()

def onClick(event):
    print('click', event.button)
    sys.stdout.flush()
    if event.inaxes:
        # display red point on all the other images where the user clicked
        print('data coords %f %f' % (event.xdata, event.ydata))
        for ax in event.canvas.figure.axes:
            ax.plot(event.xdata, event.ydata, '.', markersize=5, alpha=0.4, color='red', zorder=10)
        event.canvas.draw()
