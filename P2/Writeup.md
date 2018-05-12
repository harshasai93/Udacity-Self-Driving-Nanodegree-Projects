#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[afterAugmentation]: ./WriteUpImages/afterAugmentation.png "after Augmentation"

[afterGrayscale]: ./WriteUpImages/afterGrayscale.png "after Grayscale"

[afterNormalizing]: ./WriteUpImages/afterNormalizing.png "after Normalizing"

[beforeAugmentation]: ./WriteUpImages/beforeAugmentation.png "before Augmentation"

[beforeGrayscale]: ./WriteUpImages/beforeGrayscale.png "before Grayscale"

[FoundImagesPredictions]: ./WriteUpImages/FoundImagesPredictions.png "Found Images Predictions"

[randomAffineTransformation]: ./WriteUpImages/randomAffineTransformation.png "random Affine Transformation"

[RandomScaling]: ./WriteUpImages/RandomScaling.png "Random Scaling"

[randomTranslation]: ./WriteUpImages/randomTranslation.png "random Translation"

[SoftmaxProbabilities]: ./WriteUpImages/SoftmaxProbabilities.png "Softmax Probabilities"

[WebFoundImages]: ./WriteUpImages/WebFoundImages.png "Web Found Images"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/harshasai93/Udacity-Self-Driving-Nanodegree-Projects/blob/master/P2/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

1. Random images from the dataset with their labels on top to see how the dataset samples looks when printed on screen.
![alt text][beforeGrayscale]

2. Bar chart to show the number of samples available for each label in the dataset.

![alt text][beforeAugmentation]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces the complexity of the calculations done by the model. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][afterGrayscale]

As a last step, I normalized the image data to bring the image to normal distribution

Here is an example of a traffic sign image before and after grayscaling.

![alt text][afterNormalizing]

I decided to generate additional data because the model should be trained uniformly for all the labels to get a unbiased classification.

To add more data to the the data set, I used the following techniques because these changes tranform the image to a new image without any effect on the disinguishing features,   

1) Random scaling
![alt text][RandomScaling]

2) Random translation
![alt text][randomTranslation]

3) Random Affine Transformation
![alt text][randomAffineTransformation]

The difference between the original data set and the augmented data set is the following ... 

![alt text][afterAugmentation]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 	     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Activation			|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 	        | 1x1 stride, valid padding, outputs 10x10x16	|
| Activation 	        | 												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flatten	      		| 2x2 stride, valid padding, outputs 400 		|
| Fully Connected		| outputs 120        							|
| Activation			|												|
| dropout				| 												|
| Fully Connected		| outputs 84        							|
| Activation			|												|
| dropout				| 												|
| Fully Connected		| outputs 83        							|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the LeNet architecture with some modifications. I used the Adam optimizer with batch size 100, epochs 60, learning rate 0.009. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.946
* test set accuracy of 0.929

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The First architecture i choose was LeNet architecture as it is an established model for image classification. 
The initial architecture yielded high training-set accuary but low validation set accuracy. This meant that the model was Over-fitting the data. 
I used a well known regularization technique dropout to over-come overfitting.  
So I added two layers of drop-out after each of the Fully Connected Layer except the last one.
The Validation accuracy improved significantly to 94.6 % by this method.

After that I adjusted the Hyper-parameters like keep-prob, learning rate, epochs and batch-size for several iterations to achieve the validation accuracy of 94.6%

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
I used the Lenet architecture as a starting point as it is a well established convolutional neural network to recognize visual patterns from pixel images
with minimum pre-processing. 

LeNet was originally used for recognition of handwritten and machine printed characters, and was primarily used by post-offices. Recognition of traffic signs was 
similar to that application as the images can scaled to the same resolution as was that used for character recognition.

The traffic sign images can be represented using minimum pixel values like 32x32x3 which is perfect for Lenet. The time taken and processing power required 
will be less due to this. Also our input images size 32x32x3 which matched exactly with Lenet model described in the video lessons. 

The final model yielded training set accuracy of 1.000 , validation accuracy of 0.946 and test set accuracy of 0.929 which proves that the
model is well suited for traffic sign recognition.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][WebFoundImages] 

images 1(No entry), 2(General caution), 4(Speed limit (60km/h)) might be easier to classify as the shapes in them are relatively less complex, like staight lines or simple curves. The model was able to classify all three simple images correctly.

But images 3(Slippery road),5(Road work), 6(No passing) might be difficult to classify as the images contain more complex shapes like 
human,cars, slippery road. But the model was able to classify images 3 and 5 correctly but not the image 6.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![alt text][FoundImagesPredictions] 


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No entry   									| 
| General caution     			| General caution										|
| Slippery road					| Slippery road											|
| 	Speed limit (60km/h)	      		| 	Speed limit (60km/h)				 				|
| Road work			| Road work      							|
| No passing          | Go straight or right


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. This compares favorably to the accuracy on the test set of 92.9%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 22nd cell of the Ipython notebook.

![alt text][SoftmaxProbabilities] 

From the above the figure, the model was able to make 100% correct predictions for the first 5 images namely 'No entry','General caution', 'Slippery road', 'Speed limit (60km/h)' and 'Road work' with a probablity of 1.0.

For the last image which is a 'No passing' sign the model made an incorrect prediction of 'Go straight or right' sign also with a probability of 1.0. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


