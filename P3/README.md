# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/nVidiaModel.png "Model Visualization"
[image2]: ./output_images/normal_image.png.png "Normal Image"
[image3]: ./output_images/flipped_image.png "Flipped Image"
[image4]: ./output_images/cropped_image.png "Cropped Image"
[image5]: ./output_images/resized_image.png "Resized Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* BehavioralCloning.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Readme.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The BehavioralCloning.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the Nvidia model. 

![nvidia model][image1]

I've added the following adjustments to the model. 

- I used Lambda layers for Preprocessing image like cropping, resizing and Image Normalization
- I've added an additional dropout layer to avoid overfitting after the convolution layers.
- I've also included ELU for activation function for every layer except for the output layer to introduce non-linearity.

- Cropping
- Resizing
- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: relu
- Drop out (0.5)
- Convolution: 5x5, filter: 36, strides: 2x2, activation: relu
- Drop out (0.5)
- Convolution: 5x5, filter: 48, strides: 2x2, activation: relu
- Drop out (0.5)
- Convolution: 3x3, filter: 64, strides: 1x1, activation: relu
- Drop out (0.5)
- Convolution: 3x3, filter: 64, strides: 1x1, activation: relu
- Drop out (0.5)

- Flatten Layer

- Fully connected: neurons: 100, activation: ELU
- Drop out (0.7)
- Fully connected: neurons:  50, activation: ELU
- Drop out (0.7)
- Fully connected: neurons:  10, activation: ELU
- Drop out (0.7)
- Fully connected: neurons:   1 (output)


The below is an model structure output from the Keras which gives more details on the output shapes of each layer,

| Layer (type)                   |Output Shape      |
|--------------------------------|------------------|
|(Cropping2D)                    |(None, 160,320,3) |
|lambda_1(tf.image.resize_images)|(None, 66, 200, 3)|
|lambda_1 (Normalize)            |(None, 66, 200, 3)|
|Conv2D_1 (Convolution2D)        |(None, 31, 98, 24)|
|dropout (Dropout)               |(None, 31, 98, 24)|
|Conv2D_2 (Convolution2D)        |(None, 14, 47, 36)|
|dropout (Dropout)               |(None, 14, 47, 36)|                 
|Conv2D_3 (Convolution2D)        |(None, 5, 22, 48) |
|dropout (Dropout)               |(None, 5, 22, 48) |            
|Conv2D_4 (Convolution2D)        |(None, 3, 20, 64) |
|dropout (Dropout)               |                  |
|Conv2D_5 (Convolution2D)        |(None, 1, 18, 64) |
|dropout (Dropout)               |                  |
|flatten_1 (Flatten)             |(None, 1152)      |
|dense_1 (Dense)                 |(None, 100)       |
|dropout (Dropout)               |                  |
|dense_2 (Dense)                 |(None, 50)        |
|dropout (Dropout)               |                  |
|dense_3 (Dense)                 |(None, 10)        |
|dropout (Dropout)               |                  |
|dense_4 (Dense)                 |(None, 10)        |

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer with loss function as 'mean squared error', so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

I Used the data from the Udacity data set with some augmentation and adjustments.  

As the model was biased to straight driving, I loaded the left and right camera images adjusting the sterring angle by 0.2 to increase the sample size.

After that I removed randomly 40% of the images which had zero steering angle to reduce staright bias.

Normal Image:

![Normal Image][image2]

Also I cropped 60 pixels from the top and 25 pixels from the bottom of the image to remove the sky and front of the car.

Cropped Image:

![Normal Image][image4]

I augmented the dataset further by flipping the images with steering more than 0.3 to reduce the bias on any of the directions.

Flipped Image:

![Normal Image][image3]

Resized Image:

![Normal Image][image5]



#### 2. Creation of the Training Set & Training Process

To make the model learn the recovery behaviour, I collected data from simulator by coming to the center from the edges.

After the collection process, I had 15268 number of data points. I then preprocessed this data by cropping, resizing and normalizing the dataset. I also augmented the dataset by flippinf images.

I finally randomly shuffled the data set and put 0.2% of the data into a validation set. I used a generator to feed the training process data in batches, this reduced the load in the memory considerably and increased the training time.

The model was training with reducing loss for less than 10 epochs.

#### Final Video

Here's a link to [Project Video](./run2.mp4)

---

### Discussion

The data was the deciding factor whether the vehicle rides successfully or not. In the First attempt the vehicle due to straight bias in the data was easily colliding with edge. After removing the images with zero steering angle the performance immediately improved without changing anything. Hence the dataset should be balanced and with all possible scenarios for the model to perform the best. 

Then I collected the data for recovery from edges, which made the car driving in the center for most of the video. The only possible improvements is to collect more data from other tracks and augment more data to represent the edge scenarios.


