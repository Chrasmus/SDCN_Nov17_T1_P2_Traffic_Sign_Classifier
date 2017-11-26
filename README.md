# **Traffic Sign Recognition**

## Writeup

### Udacity Course, October 2017 cohort

**Self-Driving Car Engineer Nanodegree Program

**Project 'Traffic Sign Classifier', November 2017
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
[Training_data]: ./Plots/Distribution_in_Training_data_nov17.png "Label distribution in training data set"
[Validation_data]: ./Plots/Distribution_in_Validation_data_nov17.png "Label distribution in validation data set"
[Test_data]: ./Plots/Distribution_in_Test_data_nov17.png "Label distribution in test data set"
[Augmented_Training_data]: ./Plots/Distribution_in_Augmented_Training_data_nov17.png "Label distribution in augmented training data set"
[100_Yield_signs]: ./Plots/100_Yield_signs.png "100 examples of Yield signs from the training data set"
[Traffic_signs_for_each_label]: ./Plots/Traffic_signs_for_each_label.png "One traffic signs for each label"
[Original_sign]: ./Plots/Original_sign.png "Original sign, label = 1"
[Augmented_sign]: ./Plots/Augmented_sign.png "Augmented sign, label = 1"
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./Extra_Signs/13_Yield_Hamburg.png "Traffic Sign 1"
[image5]: ./Extra_Signs/17_No_entry_Hamburg.png "Traffic Sign 2"
[image6]: ./Extra_Signs/27_Pedestrians_Hamburg.png "Traffic Sign 3"
[image7]: ./Extra_Signs/35_Ahead_only_Hamburg.png "Traffic Sign 4"
[image8]: ./Extra_Signs/38_Keep_right_Hamburg.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Chrasmus/SDCN_Nov17_T1_P2_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

(the code can be found in [3] in the [project code](https://github.com/Chrasmus/SDCN_Nov17_T1_P2_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb))


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is bar charts showing the distributions of the labels in the training, validation, and test data sets:

Training data set
![alt text][Training_data]

Validation data set
![alt text][Validation_data]

Test data set
![alt text][Test_data]

(the code can be found in [8] and [9] in the [project code](https://github.com/Chrasmus/SDCN_Nov17_T1_P2_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb))


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At first, I decided not to do any augmentation on the training images. Instead I tried to get the model to perform better than the .75, that running the original LeNet-5 model on the untouched (but normalized) training data. After adding more depth to the two convoluted layers and also adding two Drop-outs before the fully connected layer, the model had an accuracy of up to 0.963.

Next step was add augmentated images. I copied each image in the training data set, rotate and translate it randomly, and add it to the training data. This gave me **69598** images, which provide 85% training data and 15% validation data. If I should do the split myself, I would have chosen an 80/20 pct ratio. So this way the training data matches the validation data much better than just using it as is were delivered.

I did not do any grayscaling as I am convinced that the colors are a very important feature of traffic signs. As I have done some testing on downloaded German traffic signs, with zero predictions (sigh), my afterthoughts has concentrated on how to do a better color augmentation, as this seems to be the key to unlock a better performance on new images. But I'm running out of time in this project, so it will be another time and another project, when I'll dig deep into that world.

As a last step, I normalized the image data because the model performs much better with zero-centered data, where the values fall fall within the range -1 and 1.
I did not use the suggested formula of normalizing the images using : **(pixel - 128) / 128**, because it didn't give any rise in the validations in the model. Instead I found **(pixel / 122.5) - 1** in the forums and It make the model perform much better.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I chose to use the LeNet-5 model with a few modifications. I added twice as much depth to two convoluted layer and two DropOuts to the fully connected layers, ending up with a 3x32x32-C12-MP2-C32-MP2-120N-DO-84N-DO-43N, with Relu activations

The code for making predictions on my final model is located in the 14th cell in the Jupyter notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Flatten       | Output 800
| Fully connected		| Output 120        									|
| RELU					|												|
| DropOut	      	|   				|
| Fully connected		| Output 84        									|
| RELU					|												|
| DropOut	      	|   				|
| Fully connected		| Output 43        									|
| Softmax				|         									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used these hyperparameters:

| Hyperparameter   | Value   | Comments    |
| my  | 0 | used in tf.truncated_normal to randomly define variables for the weights and biases for each layer |
| sigma | 0.1 | same as for my |
| learning rate | 0.001 | When setting the value to 0.0005, the model didn't perform very well at all (~ 0.04) |
| epochs | 10 | The validation and the loss peaked at approximately 8-9 epochs |
| batch size | 128 | the model was run on a big Mac Pro and took 22 minuttes to train |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Yield					| Speed limit (30km/h)											|
| No entry      		| Speed limit (30km/h)   									|
| U-turn     			| Speed limit (30km/h) 										|
| 100 km/h	      		| Speed limit (30km/h)					 				|
| Slippery Road			| Speed limit (30km/h)      							|


The model was not able to correctly guess any of the 5 traffic signs, which gives an accuracy of **0%**. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 24th cell in the Jupyter notebook.

In general, the model is relatively sure that all the signs Speed limit (30km/h) signs (probability between 0.33 and 0.37), but none the selected traffic signs are showing Speed limit signs.
I observed while running 15-20 models, where the signs in the training data set where shuffled before each model building, that the predictions were very much like this one, with Speed limit (30km/h) coming out as the predicted sign each time. The attempt to augment the training data by means of rotation and translation did not chance this outcome.

The top five soft max probabilities were

**1. image (Yield)**

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| .33         			    | Speed limit (30km/h)   									      |
| .07     				      | General caution 										          |
| .07					          | Children crossing											        |
| .05	      			      | Right-of-way at the next intersection					|
| .05				            | Speed limit (80km/h)      							      |


**2. image (No entry)**

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| .37         			    | Speed limit (30km/h)   									      |
| .07     			      	| Speed limit (50km/h) 										      |
| .07					          | Speed limit (80km/h)											    |
| .06	      		      	| General caution					 				              |
| .06				            | Children crossing      							          |


**3. image (Pedestrians)**

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| .36         			    | Speed limit (30km/h)   									      |
| .07				            | Children crossing      							          |
| .06	      		      	| General caution					 				              |
| .06     			      	| Speed limit (50km/h) 										      |
| .06					          | Speed limit (80km/h)											    |


**4. image (Ahead only)**

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| .36         			    | Speed limit (30km/h)   									      |
| .06					          | Speed limit (80km/h)											    |
| .07     			      	| Speed limit (50km/h) 										      |
| .05				            | Children crossing      							          |
| .06	      		      	| General caution					 				              |


**5. image (Keep right)**

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| .33         			    | Speed limit (30km/h)   									      |
| .06	      		      	| General caution					 				              |
| .05				            | Children crossing      							          |
| .07					          | Speed limit (80km/h)											    |
| .07     			      	| Speed limit (50km/h) 										      |
