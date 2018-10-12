## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Apply a Mask to extract only the lane part of the perspective transformed image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Note: Added one more point to the above goals for masking the perspective transformed image to extract only the 
lane part of the image.

[//]: # (Image References)

[image1]: ./output_images/Undistortion.jpg "Undistorted"
[image2]: ./output_images/Undistortion_Road.jpg "Road Transformed"
[image3]: ./output_images/Perspective.jpg.jpg "Perspective Transform"
[image4]: ./output_images/Magnitude_Threshold.jpg "Magnitude Thresholding"
[image5]: ./output_images/Direction_Threshold.jpg "Direction Thresholding"
[image6]: ./output_images/Combined_Thresholding.jpg "Combined Direction and Direction Thresholding"
[image7]: ./output_images/Masking.jpg "Masking"
[image8]: ./output_images/pipeline.jpg "Pipeline"
[image9]: ./output_images/Pipeline2.jpg "Pipeline2"
[image10]: ./output_images/Pipeline3.jpg "Pipeline3"
[image11]: ./output_images/Histogram_Detected_Lane_Pixels.jpg "Histogram Detected Lane Pixels"
[image12]: ./output_images/Sliding_Window.jpg.jpg "Sliding Window"
[image13]: ./output_images/Final_Output.jpg "Final Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 'getDistortionCoefficients()' method of the IPython notebook located in "AdvancedLaneFinding.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistortion][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for this step is contained in the 'undistort()' method of the IPython notebook located in "AdvancedLaneFinding.ipynb"

I used the method cv2.undistort to undistort the road image. This method takes in the input image and distortion coeeffients
obtained from the cv2.calibrateCamera method.

![Undistorted Road][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `pers_Trans()` of the IPython notebook.  The `pers_Trans()` function takes as inputs an image (`img`). Inside the function the source (`src`) and destination (`dst`) points are declared.  I chose to hardcode the source and destination points in the following manner:

```python
    src = np.float32([[215,720],[600,450],[690, 450],[1115,720]])

    dst = np.float32([[400,720],[400,0],[890,0],[890,720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 215, 720      | 400,720       | 
| 600, 450      | 400,0         |
| 690, 450      | 890,0         |
| 1115,720      | 890,720       |

I verified that my perspective transform was working as expected by plotting a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Assuming that the camera position will remain constant and that the road in the videos will remain relatively flat.

![Perspective Transform][image3]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

##### Magnitude Thresholding
I had implemented the magnitude thresholding in the method 'mag_thresh()' which takes in the input as image, orientation which can be x, y or both , kernel size and threshold minimum and maximum threshold values. 'cv2.Sobel()' method was used to calculate the gradients. both direction gradients were calculating by taking the square root of the sum of gradients in x and y direction.
I experimented with a lot of combination of values for sobel_kernel, thresh_min and thresh_max. But I ended up using the below values which gave the best results

orient='x', sobel_kernel=31, thresh_min=20, thresh_max=100
orient='y', sobel_kernel=31, thresh_min=20, thresh_max=100
orient='both', sobel_kernel=3, thresh_min=20, thresh_max=100

![Magnitude Thresholding][image4]

##### Direction Thresholding

I had implemented the magnitude thresholding in the method 'dir_threshold()' which takes in the input as image, kernel size and threshold minimum and maximum threshold values. 'np.arctan2()' method was used to calculate the direction angle to the vertical.
I experimented with a lot of combination of values for sobel_kernel, thresh_min and thresh_max. But I ended up using the below values which gave the best results

sobel_kernel=17, thresh=(0, 0.3)

![Direction Thresholding][image5]

##### Combined Thresholding

I had implemented the magnitude thresholding in the method 'dir_threshold()' which takes in the input as image, kernel size and threshold minimum and maximum threshold values.

I had combined the thresholded binary of x direction and direction threshold by logical 'and' operation to obtain binary image which had pixels detected if it was detected in both magnitude thresholding in x direction and direction thresholding. Other pixel detection are removed.

![Combined Direction and Direction Thresholding][image6]

#### Masking the region of interest

I had implemented the masking the region of interest in the method 'region_of_interest()'. This method masks out all the image except the region specified by the vertices.

![Masking][image7]

#### Pipeline(Combined Magnitude Thresholding)

This is implemented in the method 'pipeline()'.  This pipeline is made of the following steps

1. Undistortion
2. Perspective Transform
3. Combinened Magnitude and Direction Thresholding
4. Masking Lane Area

![Pipeline][image8]

#### Pipeline2(Color Thresholding)

This is implemented in the method 'pipeline2()'.  This pipeline is made of the following steps

1. Undistortion
2. Perspective Transform
3. Color Thresholding
4. Masking Lane Area
 
For Color Thresholding I extracted the R,G,B and H,S,V Channels. For detecting Yellow lines R and G channels are efficient so I combined them using logical 'AND' and combined with S channel which denotes amount of color with logical 'OR'. The result obtained is below

![Pipeline2][image9]

#### Pipeline3(Combined Magnitude and Color Thresholding)

This is implemented in the method 'pipeline3()'. This pipeline is made of the following steps

1. Undistortion
2. Perspective Transform
3. Magnitude Thresholding
4. Color Thresholding
5. Masking Lane Area

![Pipeline3][image10]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This is implemented in the method 'detect_Lanes_Histogram_c()'.
First the pipeline applied binary image is obtained and passed to the 'find_lane_pixels()' method.  This method detects the bottom of the lanes by extracted x positions of the two highest peaks in the histogram. Then a window is moved upwards determining the pixels inside the window, the window is adjusted based on the mean of the valued detected in the window.
The non-zero x and y values of the left and right lanes are fitted using 'np.polyfit()'.

![Histogram Detected Lane Pixels][image11]

![Sliding Window][image12]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is implemented in the methods 'fit_poly_c()' and draw_lane().

The formula for calculating the radius is taken from the lecture notes
```
left_curverad = ((1 + (2*left_fit_real[0]*y_eval*ym_per_pix + left_fit_real[1])**2)**1.5) / np.absolute(2*left_fit_real[0])
```

The formula for calculating the position of the vehicle with respect to center

```
np.average((left_fitx + left_fitx)/2) - binary_warped.shape[1]//2)*3.7/700)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is implemented in the method 'fit_poly_c()' methods. The Final Output of the image processing is below


![Final Output][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to [Pipeline 1 Project Video](./output_videos/pipeline_project_video.mp4)

Here's a link to [Pipeline 2 Project Video](./output_videos/pipeline2_project_video.mp4)

Here's a link to [Pipeline 3 Project Video](./output_videos/pipeline3_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Arriving at a satisfactory thresholding was the most crucial part of the project as it required more experimentation than the lane finding part. Even after lot of attempts in changing the thresholding values there were wrong detections in the shadow, road color changes of the video. Still a better method can be investigated to get the lane lines in different conditions.

The detection errors in shadow regions of the video was smoothed by taking average of the previous detections. There is scope of implementing a even robust approach when lane pixels were not detected in a frame at all then an exception may be thrown while fitting the lanes with no points. In that the case the exception should be caught and handled by using previous frames lines.

Masking and Perspective transform should be improved further as the hard coding of the region of interest, source and destination points will not work for different camera orientations and disturbances in the camera.
