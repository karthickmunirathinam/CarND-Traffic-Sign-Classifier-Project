# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./writeup_images/trafficsigns.png "Visualization of signs for each class"
[image2]: ./writeup_images/dataset_before.png "Data set distribution per class"
[image3]: ./writeup_images/dataset_after.png "Data set distribution per class after preprocessing"
[image4]: ./writeup_images/preprocess.png "Preprocessed dataset"
[image5]: ./writeup_images/test_signs.png "Test Traffic Sign "
[image6]: ./writeup_images/softmax.png "Softmax probabilities"
[image7]: ./writeup_images/Validation.png "Validation and training accuracy at every epoch"
[image8]: ./writeup_images/learningrate.png "Learning rate vs optimizer. Note that time of 120 seconds means network failed to train "
[image9]: ./writeup_images/optimizer.png "Choice of optimizer over validation accuracy "


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is a link to my [project code](https://github.com/karthickmunirathinam/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410 
* The size of test set is 12630 
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The traffic sign in every class is visualized.

![alt text][image1]

The trainng data distributution per class is shown below

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As can observe that the data are not evenly distributed in all the classes. So, I have decided to append more data to the original set. The amount of data appended depends on the mean of the datas in the classes. I have specifically choosen to rotate the image at certain set of angles and append to the original data set. We can also do translation, shering, varying contrast and adding gaussian noise of image. For simplicity, I have choosen the rotation transformation on randown angles on the angle set. This new data set now has the following class distribution.

![alt text][image3]

As a first step, I decided to convert the images to grayscale because the color characteristics does not influence much on the classification of traffic sign. The image data should be normalized so that the data has mean zero and equal variance.

Here is an example of an original image and an augmented image:

![alt text][image4]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The same architecture of LeNet neural network was used in this project with few modifications and fine tuning. After few test runs based on the accuracy, I have finalized with 2 convolution layer and 3 fully connected layer with dropouts. If there was one less convolutional layer or one less fully connected layer, I was able to observe the training was faster but the accuracy dropped. 

Adding dropout was another boost in accuracy of test data set. With drop probability of 1, I was getting to validation accuracy of around 98.2%, with test accuracy down by 89%, this was clearly sign of overfitting. Changing the drop out probability from 1 of 0.5, The test accuracy increased from 89% to 92%.

Converting from color to gray scale increased the speed of training by several folds.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image    					| 
| Convolution-1  5x5    | 1x1 stride, VALID padding, outputs 28X28X6 	|
| RELU					|												|
| Dropout				| 							                    |
|                       |                                               |
| Max pooling           | 2x2 stride,  outputs 14x14x6                  |
| Convolution-2 5x5     | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Fatten                | Output = 400.                                 |
| Dropout               |                                               |
| Fully connected-1     | Output = 120.                                 |
| RELU                  |                                               |
| Fully connected-2     | Output = 84.                                  |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected-3     | Output = 43.                                  |

The training and validation accuracy is plotted at different epoch is shown below. A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting. We can observe the we have minimal difference between the training and the validation set in the higher epochs.

![alt text][image7]
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
The learning rate is choosen and tuned with the model. The choice of learning rate also depends on the type of optimizer we use. This can be depicted in the figure below. The choice of learning rate affects the training rate by several magnitudes. There is a valley shape for each optimizer: too low a learning rate never progresses, too high a learning rate causes instability and never converges. In between there is a band of "just right" learning rates that successfully train.

![alt text][image8]

Adam optimizer is more stable than the other optimizers, it doesn’t suffer any major decreases in accuracy. Adams also learns faster. The below figure shows the comparison of various optimizers with Validation accuracy vs time.

![alt text][image9]

Regarding the batch size the choosen based on how the model seems to perform and I found 128 was optimal. The epoch was choosen incrementally until the value where the  accuracy is not changing considerably and increased time taken for training. I have choosen 20 epochs based on this observation. According to the therory from it was adviced to choose the zro  mean and small standard deviation. Very low standard deviation was giving no big difference, where are higher standard deviation leads to more training time.

To train the model, I have tuned the hyper parameters to maximize the validation accuracy. After several trails the hyperparameter I decided is as follows.
* Learning rate is 0.001
* Epochs is 20
* Batch size is 128
* Adams Optimizer for better accuracy.
* Mean for Weights matrix initialization is 0
* Standard Deviation for Weights matrix initialization is 0.1
* dropout probability of 0.5


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 97.8%
* validation set accuracy of 97% 
* test set accuracy of 92%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image5]

The test accuracy is 90% as one of the sign is incorrectly classified. The reason is due to noise which lead to incorrect classification. I have also applied few traffic sign with more noise interms of contrast and having multiple signs which lead to inaccurate prediction. Let us take an example of the case where the "caution" sign is misclassified as "turn right ahead". This is due to the influence of background and the secondary sign describing the text in German. We have not trained the network based such complicated signs which leads to misclassification. The way we can overcome this issue is make model more robust by more training data with several scenarios involving background, multiple signs, jittered, sheared, scaled and with different lighting conditions. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Go straight or left   | Go straight or left 							|
| Caution				| Turn right ahead								|
| 50 km/h	      		| 50 km/h					 				    |
| 60 km/h			    | 60 km/h      							        |
| Priority Road   	    | Priority Road  								| 
| Turn left ahead       | Turn left ahead 							    |
| Road work				| Road work      								|
| No entry	      		| No entry					 				    |
| Right of way		    | Right of way      							|

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
 The softmax probabilities for each prediction of the 10 images taken from internet is shown below.
 The model is close to 100% certain of 8 out of 10 of the signs I gave it. Even on the 6th image, it's 83% certain of its prediction. However for the misclassified case the confidenc eis also high due to the reasons I mention previosly. The models classifies well most of the scenarios where the signs have less distortions and background effects.

 ![alt text][image6]



