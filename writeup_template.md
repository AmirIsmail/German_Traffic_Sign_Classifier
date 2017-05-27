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

[image1]: ./Train_Histogram.png "Visualization"
[image2]: ./test images/60_kmh.jpg "60 km sign"
[image3]: ./test images/left_turn.jpeg "Turn left sign"
[image4]: ./test images/road_work.jpg "road work sign"
[image5]: ./test images/stop.jpg "stop sign"
[image6]: ./test images/yield_sign.jpg "yield_sign"
[image7]: ./100km-grey.png "100km sign greyscaled"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

1. Histogram for the distribution of the training dataset vs output classes is plotted
2. A random image from the training dataset is plotted

./Train_Histogram.png "Visualization"

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
Steps added as a preprocessing for the data are as following:
		a. Converting images into greyscale with the help of cv python library --> This step will make it easy for the model to train as it decreases the channels of the image to 1 
		instead of 3
		b. normalizing the colours to be between 0.1 and 0.9 instead of 0 to 255 --> this step is done so that the data has mean zero and equal variance using the technique "pixel 255 * 0.8 + 0.1"

-- A visualization of the training images after preprocessing is done by plotting a random image from the training dataset "./100km-grey.png"



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation for layer1							|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					| Activation for layer2							|
| Pooling input			| 2x2 stride, valid padding Output = 5x5x16		|
| Fully connected		| Output = 400       						    |
| RELU					| Activation for the first connected layer    	|
| Dropout 				| to prevent overfitting						|
| Fully connected		| Output = 43       						    |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used the following:
	1. Adam optimizer from the tensorflow library 
	2. softmax cross entropy from tensorflow library to smooth the loss operation calculations that will then be used in analyzing the performance of the model
	3. I have chosen the learning rate to be 0.005 after testing other values and i checked that this value will not result in network overfitting

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.968
* validation set accuracy of 0.96839 
* test set accuracy of 0.89398


If a well known architecture was chosen:
* What architecture was chosen?
the architecture chosen for this project is the Lenet Architecture.
* Why did you believe it would be relevant to the traffic sign application?
I belive that Lenet is relevent to this project becasuse Lenet deals with image classifications which is what this project uses.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
With a validation accuracy of .968 and a test accuracy of .89, this was proof enough that the model is working well.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

[image2]: ./test images/60_kmh.jpg "60 km sign"
[image3]: ./test images/left_turn.jpeg "Turn left sign"
[image4]: ./test images/road_work.jpg "road work sign"
[image5]: ./test images/stop.jpg "stop sign"
[image6]: ./test images/yield_sign.jpg "yield_sign"

The first image might be difficult to classify because this image was never shown to the network while training

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km     		    | yield      									| 
| left turn    			| keep right 									|
| road work				| general caution								|
| stop      		    | stop		        			 				|
| yield		         	| yield               							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is because the chosen images from the web were not of the same resolution as the training dataset.

the first prediction had a certantity of 91% yet got the prediction completely wrong
the second prediction had a certantity of 93% yet also got the prediction wrong
the third prediction had a certantity of 98% yet got it wrong as well
the fourth prediciton had a certantity of 60% and the prediciton was correct
the fith predicion had a certatntiy of 99% and also was correct



####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a yield sign (probability of 0.13), and the image does contain a yield sign. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .13        			| yield 							    		| 
| .11     				| Stop    										|
| .11					| keep right		    						|
| .05	      			| general caution   			 				|
| .04				    | yield             							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


