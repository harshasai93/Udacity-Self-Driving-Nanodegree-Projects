## Vehicle Detection Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG_features.jpg
[image2]: ./output_images/DatasetExample.jpg
[image3]: ./output_images/spatial_binning.jpg
[image4]: ./output_images/Color_Histogram.jpg
[image5]: ./output_images/combined_features.jpg
[image6]: ./output_images/pipelineBeforeCropping.jpg
[image7]: ./output_images/pipelineAfterCropping.jpg
[image8]: ./output_images/pipeline_After_Multi_sized_window.jpg
[image9]: ./output_images/Windows.jpg
[image10]: ./output_images/HeatMap.jpg
[video1]: ./output_videos/project_video16.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained under the heading 'Extracting Features' in the Jupyter Notebook 'Vehicle_Detection.ipynb'

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image2]

I then extracted features from a sample image from the dataset. I extracted Spatial Binning, Color Histogram and HOG features of the image. The plots of the features are below 

1. Spatial Binning

![alt text][image3]

2. Color Histogram

![alt text][image4]

3. HOG Features 

![alt text][image1]

4. Combined Features

![alt text][image5]

#### 2. Explain how you settled on your final choice of HOG parameters.

First I tried by creating the features of all three Spatial Binning, Color Histograms and HOG Features with the given parameters. 

| Configuration Label | color_space   |  orient  |  pix_per_cell | cell_per_block | hog_channel |  spatial_size |  hist_bins|
|:-------------------:|:-------------:| :-------:| :------------:| :-------------:| :----------:| :------------:|:---------:|
| 1                   | RGB           |     9    |     8         |     2          |      ALL    |   (16, 16)    |     16    |

Next I removed Spatial Binning, Color Histograms from the features to train only with HOG features to reduce the feature extraction and training time,

| Configuration Label | color_space   |  orient  |  pix_per_cell | cell_per_block | hog_channel |  spatial_size |  hist_bins|
|:-------------------:|:-------------:| :-------:| :------------:| :-------------:| :----------:| :------------:|:---------:|
| 2                   | RGB           |     9    |     8         |     2          |      ALL    |      N/A      |     N/A   |
| 3                   | HSV           |     9    |     8         |     2          |      ALL    |      N/A      |     N/A   |
| 4                   | HLS           |     9    |     8         |     2          |      ALL    |      N/A      |     N/A   |
| 5                   | YCrCb         |     9    |     8         |     2          |      ALL    |      N/A      |     N/A   |
| 6                   | YCrCb         |     12   |     16        |     2          |      ALL    |      N/A      |     N/A   |


The feature-vector length, Test Accuracy, Training Time, Prediction Time for random 1000 images from th test set and number of correct predictions out of them are tabulated below,

| Configuration Label |Feature vector length| Test Accuracy | Training Time | Prediction Time(1000) | Correct Predictions(1000) |
|:-------------------:|:-------------------:| :------------:| :------------:|:---------------------:| :------------------------:|
| 1                   |        6108         |     0.982     |     17.27     |         6.54          |          995              |
| 2                   |        5292         |     0.9718    |     17.4      |         7.72          |          992              |
| 3                   |        5292         |     0.9803    |     14.82     |         6.79          |          996              |
| 4                   |        5292         |     0.9783    |     15.97     |         7.55          |          992              |
| 5                   |        5292         |     0.9803    |     2.9       |         7.54          |          997              |
| 6                   |        1296         |     0.9848    |     2.02      |         3.04          |          996              |

I got Test Accuracy of 98.2 in the first attempt with all three features without tuning the parameters at all. But I wanted to reduce the Training time and the time for prediction. 

After some iterations on different color spaces and parameter tuning I got very good result with Test Accuracy of 98.48, Training Time of 2.02 seconds and Prediction time of 3.04 seconds for 1000 images.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained under the heading 'Training the SVM Classifier' in the Jupyter Notebook 'Vehicle_Detection.ipynb'.

I trained a linear SVM using the method LinearSVC() from module sklearn.svm. Before training the SVM, I normalize, shuffled  features, also split the dataset for training and testing in the ratio 80:20.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained under the heading 'Sliding Window to detect cars in image using SVM' in the Jupyter Notebook 'Vehicle_Detection.ipynb'.

First I started with only 96x96 window with 50% overlap over all the image. The result is below, 

![alt text][image6]

Then I only searched for the bottom half of the image below [350, 638] . The result is below,

![alt text][image7]

I went ahead with this window pattern, but after running it on the video I observed that the window size was too small the vehicles which are near so I implemented a multi sized sliding window depending on the distance from the our car. The result is visualized below,

![alt text][image9]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using YCrCb 3-channel HOG features with orient = 12, pix_per_cell = 16 and cell_per_block = 2, which provided a nice result. I first reduced the size of the feature vector, then I experimented with color spaces to decide finally that YCrCb was the best color-space. Also increased the orient to 12 and pix-per_cell to 16.

Here are some example images. As the sliding windows dimensions will be the same for any frame of the image, I created a method getWindows() which will be calculated once and passed to movie.py to detect the hot windows where the classifier predicts for cars.

![alt text][image8]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video16.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained under the heading 'Heat Map' in the Jupyter Notebook 'Vehicle_Detection.ipynb'.

I created a deque object of maxlen=30 to hold the detection from frames. Then I recorded the positions of positive detections in each frame of the video and added them to the deque object. Then from several positive detections stored in the deque object I created a heat map and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a heat map of a frame

![alt text][image10]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The HOG features was the only features in the feature vector used to train the SVC. I feel there may be more useful features to detect vehicles which must be further researched to achieve a better result. Also the data set can be further improved to include all the variations of the vehicles. Also different classes of data like pedestrians, bikes, trucks can be added for predictions on the road. That would be amazing to try out!

This pipeline does a good job of finding approximate positions of cars, but that is not accurate enough for self-driving cars. The bounding boxes should be more tight, which means that multi-sized sliding window search can be improved. 

The Heatmap is dependent on the cache-size and thresholding limit for detecting vehicles, but some false-positives may slip through the thresholding wall, so is a area to work on further. As it is a multi-step algorithm improving one step can solve the other step's problems. If the classifier and sliding window is improved, it may solve the threshold leak problem. I spent a lot of time on tuning the cache size and threshold limit value which may be due to the classifier and sliding window problems.
