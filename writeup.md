# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./output_images/train_dataset_label_distribution.jpg "Train"
[image2]: ./output_images/validation_dataset_label_distribution.jpg "Validation"
[image3]: ./output_images/test_dataset_label_distribution.jpg "Test"
[image4]: ./output_images/sample_traffic_sign.jpg "Sample Traffic Sign"
[image5]: ./output_images/sample_traffic_sign_grey.jpg "Traffic Sign Greyscale"
[image6]: ./new_images/sign1.jpg "Traffic Sign 1"
[image7]: ./new_images/sign2.jpg "Traffic Sign 2"
[image8]: ./new_images/sign3.jpg "Traffic Sign 3"
[image9]: ./new_images/sign4.jpg "Traffic Sign 4"
[image10]: ./new_images/sign5.jpg "Traffic Sign 5"
[image11]: ./new_images/sign6.jpg "Traffic Sign 6"
[image12]: ./new_images/sign7.jpg "Traffic Sign 7"
[image13]: ./new_images/sign8.jpg "Traffic Sign 8"
[image14]: ./new_images/sign9.jpg "Traffic Sign 9"
[image15]: ./new_images/sign10.jpg "Traffic Sign 10"
[image16]: ./new_images/sign11.jpg "Traffic Sign 11"
[image17]: ./new_images/sign12.jpg "Traffic Sign 12"
[image18]: ./new_images/sign13.jpg "Traffic Sign 13"
[image19]: ./new_images/sign14.jpg "Traffic Sign 14"
[image20]: ./new_images/sign15.jpg "Traffic Sign 15"
[image21]: ./new_images/sign16.jpg "Traffic Sign 16"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set in code cell number 2 in Jupyter Notebook:

* The size of training set is: 34799
* The size of the validation set is 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: 32x32x3
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

I've compared class distribution(in percentage) of training, validation and test datasets using bar chart. (in cell number 5,6 and 7 in Jupyter Notebook)

![alt text][image1]
![alt text][image2]
![alt text][image3]

Training, validation and test datasets have similar distributions. This is essential for Learning Model to perform well. However the labels does not have similar distributions. Ideally, the labels should have similar distribution for better performance of the model.

It is possible to calculate the number of images required for each label to get the same distribution of labels for each dataset in further studies.

### Design and Test a Model Architecture

#### 1.Pre-processing Datasets

As a first step, I decided to convert the images to grayscale because I can classify than with my bare eyes without considering color of the traffic sign. If I use RGB values, my model will try to learn the color values of images too, which will increase the complexity of my model.

I used Computer Vision library (CV2) to convert RGB images to Greyscale images.
I created a function `rgbtogray` in code cell 10 in Jupyter Notebook to convert an RGB dataset to a Greyscale dataset.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4] ![alt text][image5]

As a last step, I normalized the image data because of numerical stability.
Numerical stability requires variables to have zero mean and equal variance whenever possible.  

#### 2. Model architecture

My final model consisted of the following layers:
The model can be found in code cell 14 in Jupyter notebook.

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 GREYSCALE image   						   	    |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x15 	|
| RELU					    |												                        |
| Max pooling	      | 2x2 stride,  outputs 14x14x15 				        |
| Convolution 5x5	  | 1x1 stride, valid padding, outputs 10x10x30   |
| RELU					    |												                        |
| Max pooling	      | 2x2 stride,  outputs 5x5x30				            |
| Flatten           | outputs 750                                   |
| Fully connected		| outputs 300        									          |
| RELU				      |             									                |
|	Dropout					  | keep probability: 0.8												  |
| Fully connected		| outputs 150        									          |
| RELU				      |             									                |
| Fully connected		| outputs 43        									          |

#### 3. Hyperparameters

To train the model, I used an AdamOptimizer with learning rate of 0.001

I used 50 epochs with the batch size of 64.

#### 4. Solution

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.961
* test set accuracy of 0.939

I used an iterative approach to solve the problem:
* I used Lenet architecture at first. Lenet architecture is simple and easy to apply. Before applying any complex models, I wanted to check whether it can be possible to get an accuracy higher than %93.
* I get a maximum accuracy around 0.9 even I tried increasing epochs or reducing batch sizes.
* As training accuracy higher than validation and test accuracy the model is clearly overfitting.
* I have increased the depth of images from 6 to 15 to increase the depth of the model since traffic signs are more complex than handwritings. I also decreased batch size to increase accuracy of the model. I increase number of epochs as I've seen accuracy is increasing with number of epochs.
* I choose CNN's because it doesn't matter where is the traffic sign on the image.(translation invariance). Instead of learning every image completely, we can learn a patch of it and patches can share the weights. A dropout layer helped creating a successful model because network does not rely on any given activation making model more robust and reduce overfitting.

### Testing Model on New Images

Here are sixteen German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10]![alt text][image11]![alt text][image12]![alt text][image13]
![alt text][image14]![alt text][image15]![alt text][image16]![alt text][image17]
![alt text][image18]![alt text][image19]![alt text][image20]![alt text][image21]

The images with complex backgrounds like 1st, 4th, 6th and 14th images will be difficult to classify.

The image number 11 will be difficult to classify because photo is taken from different angle.

#### 2. Model's Predictions on New Traffic Signs

Here are the results of the prediction:

| Image			    |Prediction|
|:---------------------:|:---------------------------------------------:|
| Speed limit (30km/h)                  | Speed limit (30km/h) |
| Speed limit (70km/h)                  | Speed limit (70km/h) |
| Bumpy road					                  | Bumpy road					 |
| Wild animals crossing	                | Right-of-way at the next intersection	|
| Go straight or left   	              | Go straight or left|
| Keep left                             | No entry|
| Priority road     		                | Priority road|
| Roundabout mandatory	                | Roundabout mandatory|
| Road work	                            | Road work|
| Right-of-way at the next intersection | Right-of-way at the next intersection|
| Slippery road                         | Wild animals crossing     	|
| Turn right ahead    		              | Turn right ahead 				|
| Yield					                        | Yield					|
| Stop	                                | Stop		|
| Speed limit (100km/h)	                | Speed limit (100km/h) |
| No entry	                            | No entry |

The model was able to correctly guess 13 of the 16 traffic signs, which gives an accuracy of 81%. It is lower than test set accuracy of %93.4

#### 3. Predictions

The code for making predictions on my final model is located in the 29th cell of the Ipython notebook.

For the first image, the model very sure that this is a Speed limit (30km/h) sign (probability of 1.0), and the image does contain a  Speed limit (30km/h) sign. The top five soft max probabilities were:

| Probability  |  Prediction |
|:---------------------:|:---------------------------------------------:|
| 1.00   | Speed limit (30km/h) |
| .00    | Speed limit (20km/h)	|
| .00		 | Speed limit (80km/h)	|
| .00	   | Speed limit (50km/h)	|
| .00		 | Speed limit (60km/h) |


For the second image, the model very sure that this is a Speed limit (70km/h) sign (probability of 1.0), and the image does contain a  Speed limit (70km/h) sign. The top five soft max probabilities were:

| Probability  |  Prediction |
|:---------------------:|:---------------------------------------------:|
| 1.00   | Speed limit (70km/h) |
| .00    | Speed limit (20km/h)	|
| .00		 | Speed limit (30km/h)	|
| .00	   | Speed limit (50km/h)	|
| .00		 | Speed limit (60km/h) |

For the third image, the model very sure that this is a Bumpy road sign (probability of 1.0), and the image does contain a Bumpy road sign. The top five soft max probabilities were:

| Probability  |  Prediction |
|:---------------------:|:---------------------------------------------:|
| 1.00   | Bumpy road             |
| .00    | Wild animals crossing	|
| .00		 | Slippery road	        |
| .00	   | Keep left              |
| .00		 | Bicycles crossing      |

For the fourth image, the model very sure that this is a Right-of-way at the next intersection (probability of 0.99), however the image does contain a Wild animals crossing sign. The top five soft max probabilities were:

| Probability  |  Prediction |
|:---------------------:|:---------------------------------------------:|
| .99    | Right-of-way at the next intersection  |
| .01    | Children crossing 	|
| .00		 | Beware of ice/snow	|
| .00	   | Slippery road	|
| .00		 | Dangerous curve to the right |

I expected to see Wild animals crossing sign as one of the five highest probabilities and probability of Right-of-way at the next intersection is to be less. The model is clearly overfitting.

For all other new images, the probability of selected class is 1.00 except following images:

For the eleventh image, the model pretty sure that this is a Beware of ice/snow sign (probability of 0.99), however the image does contain a Slippery road sign. The top five soft max probabilities were:

| Probability  |  Prediction |
|:---------------------:|:---------------------------------------------:|
| .99   | Wild animals crossing |
| .01   | Road work	|
| .00	 | Speed limit (60km/h) |
| .00	 | Slippery road	|
| .00	 | Dangerous curve to the left |
