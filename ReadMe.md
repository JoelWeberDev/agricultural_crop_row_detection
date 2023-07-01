# Robust Crop Row Detector

This system is designed to preform live detection of crop rows in fields with various types of crops. I has been designed to be robust in its capacity to differentiate weeds from the desired crop. The intended application of this product to modernize the 1997 [Sukup Slide Guide](https://drive.google.com/file/d/1C9tRUihWTYV-mEKkKOlWpIxzGHnaR0qW/view?usp=sharing) tractor attachment, however the final product will be designed to allow for easy integration into other agricultural applications.

### Developed By Joel Weber for Living Acres Agrobotics

## Project Details 
Since the project is at the moment incomplete I have broken the project into 3 components as they may interest a contributor or viewer

***Disclaimer*** all the software here is provided as-is and may not be fully tested at the moment. 

***Updates*** Continual developments will periocically be made to the software as the product develops


## Current Documenation
  *Description* This section includes some examples of the model working on images and live video on multiple different crops and stages of growth. It also includes the issues that 
  are evident in these working examples along with methods that are being impelemented to resolve those problems. 

  ### Early Winter Wheat
  **Analysis** Winter wheat samples like this have been the primary source of testing data up until this point. This demonstrates that the system is able to speparate rows that are 
  in close proximity to eachother. In some of the samples have had stochasitic noise introduced to simulate variance in the envoronment or moderate weed foliage in the field. All the 
  wheat in these images are spaced at an average of 18 inches
  [![Detection on winter wheat with mask and noise](Demonstration_data\Readme_images\winter_wheat_1_mask_noise.jpg)](https://youtu.be/lmLJby_kZS8)
  [![Detection on winter wheat without mask and noise](Demonstration_data\Readme_images\winter_wheat_1.jpg)](https://youtu.be/QxzcWDdI4Ac)


  ### Early Corn
  **Analysis** This here are some tests that were preformed on corn that was planted at 80 cm spacing. This test the models ability for precision when the plants are small and ability to detect when there are few rows in the frame. 

 ### Camera Data
  All test data has been gathered with a DJI mini se camera flying at 1 meter above the fields.
  To calibrate for a different camera execute the following steps:
   1. Gather 5 images of a chess/checker board with your camera
   2. In the Camera_Calib folder create a new folder named Chess_imgs where you will place the chess board images you collected 
   3. In the calibrate function of Chess_board_calib.py set square_size= the width of 1 chess board square in mm, width= the horizontal number of inner squares (not touching the board edge), height= the vertical number of horizontal sqaures 
   4. Navigate to the Camera_Calibration.py file and run it which will automatically update the Cam_spect_dict.json 


 ### Status
  The models that have been made are currently being used with prerecorded video or images and are not yet ready to handle live testing. 
  
  **Model Outline**
   1. The images or frames from the input folder are opened with the cv2.imread function.
   2. The frame is then resized, blurred, and converted to a hsv color format.
   3. In the edge_detection module the green colors are extracted from the image through the cv2.inRange function.
   4. This mask is then subdivided into multiple masks which define a more strict range of green shades.
   5. The [cv2.HoughlinesP](https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb) function is preformed on each of the masked images including the original.
   6. The approxomate vanishing point of the image is calculated and the lines are filtered by their derivative to discard any that do not pass near enough to the vanishing point.
   7. The remaining lines are grouped by proximity and derivative.
   8. All the grouped lines from every mask are gathered processed to determine where the most probable location of the crop rows are based on the the distribution of the lines.
   9. These rows are returned as the detected rows.
  The software is currently under construction and may require some adaptation to achieve correct preformance 
 ### Next Steps 
  **Adaptive Paramters**
   We are currently working to bring every hard coded value for a wide array of functions into one place that will calculate them relative to the specific nuances that may be present in the environment. This adaptive parameter model will also double as the hub for the machine learning aspect of the model where the optimal set of values to input into the function can be determined automatically.
   
  **CUDA enabled for GPU**
   The model is currently running too slow to preform real time row detection. To solve this we are working to enable this project with [Nvidia Cuda](https://developer.nvidia.com/cuda-zone#:~:text=CUDA%C2%AE%20is%20a%20parallel,harnessing%20the%20power%20of%20GPUs) to allow it to be run with a gpu for speed accelerations of 10x. 

 **General Maintainence**
  - In the code there are many remnants from old code that needs to be removed and simplified.
  - We are developing a main file that will house any value that may need to be regularily adjusted.
  - C++: The whole process is going to rewritten in C++ to boost speed and prefomance. The python that is being used now is soley for the purposes of prototyping
  - When main is added there will be an I/O interaction to set user values that are needed. (Right now they are just hidden in Adaptive_values.json)

 ### Visions
  **Stereo Vision**:
   The final objective is to develop this system as module that can be installed on the [OAK-D-CM4](https://docs.luxonis.com/projects/hardware/en/latest/pages/DM1097.html) stereo vision camera. This will allow for the 3D processing of the field to work in parallel with the color analysis. 


## Contibutuions
Pull request are welcome. Contact me at joelweberdevelopment@gmail.com to express any comments or concerns.

