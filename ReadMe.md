# Robust Crop Row Detector

This system is designed to perform live detection of crop rows in fields with various types of crops. It has been designed to be robust in its capacity to differentiate weeds from the desired crop. The intended application of this product to modernize the 1997 [Sukup Slide Guide](https://drive.google.com/file/d/1C9tRUihWTYV-mEKkKOlWpIxzGHnaR0qW/view?usp=sharing) tractor attachment, however, the final product will be designed to allow for easy integration into other agricultural applications.

### Developed By Joel Weber for Living Acres Agrobotics

## Project Details 
Since the project is at the moment incomplete I have broken the project into 3 components as they may interest a contributor or viewer

***Disclaimer*** All the software here is provided as-is and may not be fully tested at the moment. 

***Updates*** Continual developments will periodically be made to the software as the product develops


## Current Documentation
  *Description* This section includes some examples of the model working on images and live video on multiple different crops and stages of growth. It also includes the issues that 
  are evident in these working examples along with methods that are being implemented to resolve those problems. 

  ### Early Winter Wheat
  **Analysis** Winter wheat samples like this have been the primary testing data source up until now. This demonstrates that the system is able to separate rows that are 
  in close proximity to each other. Some of the samples have had stochastic noise introduced to simulate variance in the environment or moderate weed foliage in the field. All the 
  wheat in these images is spaced at an average of 18 inches. 
  </br>
  **Short clips of the row detection working** 
  
  [![Detection on winter wheat with mask and noise](https://github.com/JoelWeberDev/agricultural_crop_row_detection/blob/main/Demonstration_data/Readme_images/Crop_recog_face_image.jpg)](https://youtu.be/QxzcWDdI4Ac)
  
  [![Detection on winter wheat without mask and noise](https://github.com/JoelWeberDev/agricultural_crop_row_detection/blob/main/Demonstration_data/Readme_images/Crop_recog_face_image_masked.jpg)](https://youtu.be/lmLJby_kZS8)


  ### Early Corn
  **Analysis** Here are some tests that were performed on corn that was planted at 80 cm spacing. This test the model's ability for precision when the plants are small and its ability to detect when there are few rows in the frame. 

 ### Color Layer Approach:
 **Description** The model is currently using a series of layers that progressively separate all the pixels that lie within a specific sliver of the green color hue. A line detection algorithm is performed on one each of these layers independently. The lines from each of these layers are brought together into one frame where they are filtered by line gradient and the areas of the image that have the highest density of lines are identified as the next probable crop rows within the image frame. *Note* This will be the approach that is used with stereo vision except instead of detection being applied to zones of green pixels the heights in a [topographic](https://en.wikipedia.org/wiki/Topographic_map) frame constructed by the stereo camera will be used. 
 #### Layered Sample Without Line Grouping 
 ![Non-grouped line frame](https://github.com/JoelWeberDev/agricultural_crop_row_detection/blob/main/Demonstration_data/Readme_images/No_line_grouping.png)

 #### Layered Sample With Line Grouping (Uses many more layers which increases detection quality, but decreases speed)
 ![Grouped line frame](https://github.com/JoelWeberDev/agricultural_crop_row_detection/blob/main/Demonstration_data/Readme_images/Good_detec.png)
 

 ### Camera Data
  All test data has been gathered with a DJI mini se camera flying at 1 meter above the fields.
  To calibrate for a different camera execute the following steps:
   1. Gather 5 images of a chess/checkerboard with your camera
   2. In the Camera_Calib folder create a new folder named Chess_imgs where you will place the chessboard images you collected 
   3. In the calibrate function of Chess_board_calib.py set square_size= the width of 1 chessboard square in mm, width= the horizontal number of inner squares (not touching the board edge), height= the vertical number of horizontal squares 
   4. Navigate to the Camera_Calibration.py file and run it which will automatically update the Cam_spect_dict.json 


 ### Status
  The models that have been made are currently being used with prerecorded videos or images and are not yet ready to handle live testing. 
  
  **Model Outline**
   1. The images or frames from the input folder are opened with the cv2.imread function.
   2. The frame is then resized, blurred, and converted to a Hsv color format.
   3. In the edge_detection module the green colors are extracted from the image through the cv2.inRange function.
   4. This mask is then subdivided into multiple masks which define a more strict range of green shades.
   5. The [cv2.HoughlinesP](https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb) function is performed on each of the masked images including the original.
   6. The approximate vanishing point of the image is calculated and the lines are filtered by their derivative to discard any that do not pass near enough to the vanishing point.
   7. The remaining lines are grouped by proximity and derivative.
   8. All the grouped lines from every mask are gathered and processed to determine where the most probable location of the crop rows based on the distribution of the lines.
   9. These rows are returned as the detected rows.
  The software is currently under construction and may require some adaptation to achieve the correct performance.

 ### Next Steps 
  **Adaptive Paramters**
   We are currently working to bring every hard-coded value for a wide array of functions into one place that will calculate them relative to the specific nuances that may be present in the environment. This adaptive parameter model will also double as the hub for the machine learning aspect of the model where the optimal set of values to input into the function can be determined automatically.
   
  **CUDA enabled for GPU**
   The model is currently running too slow to perform real-time row detection. To solve this we are working to enable this project with [Nvidia Cuda](https://developer.nvidia.com/cuda-zone#:~:text=CUDA%C2%AE%20is%20a%20parallel,harnessing%20the%20power%20of%20GPUs) to allow it to be run with a GPU for speed accelerations of 10x. 

 **General Maintenance**
  - In the code, there are many remnants from old code that needs to be removed and simplified.
  - We are developing a main file that will house any value that may need to be regularly adjusted.
  - C++: The whole process is going to be rewritten in C++ to boost speed and performance. The Python that is being used now is solely for the purposes of prototyping
  - When main is added there will be an I/O interaction to set user values that are needed. (Right now they are just hidden in Adaptive_values.json)

 ### Future Plans for Development
  **Stereo Vision**:
   The final objective is to develop this system as a module that can be installed on the [OAK-D-CM4](https://docs.luxonis.com/projects/hardware/en/latest/pages/DM1097.html) stereo vision camera. This will allow for the 3D processing of the field to work in parallel with the color analysis. 
   **Machine Learning for Row Recognition**
   The plan is to release an initial testing prototype of the project that will gather data while in use. The data that is collected will be used to train special classifier models tailored to the detection of the crop rows. This development has the potential to massively boost performance and decrease the degree of manual programming that will be required.
   


## Contibutuions
Pull requests are welcome. Contact me at joelweberdevelopment@gmail.com to express any comments or concerns.

