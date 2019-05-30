# **Traffic Sign Recognition** 

## Writeup


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
[image0]: ./writeup_images/data_explore.png "data"
[image1]: ./writeup_images/bar_train.png "barchart"
[image1a]: ./writeup_images/bar_test.png "barchart"
[image1b]: ./writeup_images/bar_valid.png "barchart"
[image2]: ./writeup_images/grayscale.png "Grayscaling"
[image3]: ./writeup_images/norm_images.png "Norm"
[image4]: ./new_images/11_Rightofway.jpg "11:-Rightofway"
[image5]: ./new_images/12_PriorityRoad.jpg "12 :-PriorityRoad"
[image6]: ./new_images/14_Stop.jpg "14:-Stop"
[image7]: ./new_images/17_Noentry.jpg "17:-Noentry"
[image8]: ./new_images/25_RoadWork.jpg "25:-RoadWork"
[image9]: ./new_images/33_RightOnly.jpg "33:-RightOnly.jpg"
[image10]: ./writeup_images/softmax_prob1.PNG
[image11]: ./writeup_images/softmax_prob2.PNG
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/7ee8d0d4-561e-4101-8615-66e0ab8ea8c8/concepts/a96fb396-2997-46e5-bed8-543a16e4f72e)

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

Here is an exploratory visualization of the data set. It is a bar chart showing how the data feature class is distributed across the training,testing and validation dataset.sample images can also be seen in the html present in the directory.
##### sample data
![alt text][image0]
##### training set
![alt text][image1]
##### testing set
![alt text][image1a]
##### validation set
![alt text][image1b]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it preserves the features by reducing the size of the image by reducing the channels to 1 from RGB scale.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because normalization helps to keep the pixel values within [0 1]or [-1 1] range by scaling the features also it can be used to keep the mean as 0 and standard deviation as 1 across the data.I have tried many methods for preprocessing the data :

    (a)Method1 :-scaling to range [-1,1] by subtracting and dividing pixel values by 128.
    
    (b)Method2 :-Standardisation with mean and standard deviation to make data centred around 0 mean and 1 std deviation.
    
    (c)L2 Normalization: Vector is divided by the magnitude of the vector to give L2 norm.
    
    (d)Image Negative: Negaative of image is taken to highlight important features.
    
    (e)L1 Normalization: Vector is divided by absolute value sum of vector elements.
out of these methods,I found method 1 and L2 norm performing better compared to other.
##### Normalized
![alt text][image3]

I decided to generate additional data because dnn requires large data for training and it also helps for regularization 

To add more data to the the data set, I used the keras image augmentation method where i have introduced ZCA and image shift to generate fake data but since the training takes considerable amount of time and even without additional data my model was giving more than 93 percent in validation set,I thought of keeping it just for value add purpose.

The difference between the original data set and the augmented data set is the following ZCA whitening and shifting. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					      | 
|:---------------------:|:---------------------------------------------------:| 
| Input         		| 32x32x1 gray image   							      | 
| Convolution 5x5     	| 1x1 stride,6 filters,valid padding, outputs 28x28x6 |
| RELU					| activation									      |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				      |
| Convolution 5x5	    | 1x1 stride,16 filters,valid padding,outputs 10x10x16|
| RELU					| activation									      |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				      |
| Convolution 5x5	    | 1x1 stride,412 filters,valid padding,outputs 1x1x412|
| RELU					| activation									      |
| Flatten				| input-412	tensor								      |
| Fully connected		| hidden layers-122,dropout-0.5        			      |
| Fully connected		| hidden layers-84,dropout-0.5        			      |
| Output layer  		| layers-43                          			      |
| Softmax				| class probabilities      						      |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, 

| Parameter        		|     value     	        					| 
|:---------------------:|:---------------------------------------------:| 
| Optimizer        		| Adam Optimizer     							| 
| batch size        	| 32                                          	|
| Epochs     	      	| 50                            				|
| dropout       	    | 0.5       									|
| learning rate 		| 0.001        									|
| loss function			| cross entropy   								|
|						|												|
|						|												|
 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.958
* test set accuracy of 0.941

The code is present in 25 th code block after training block.

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

    I tried with Lenet architecture mentioned in course itself as a starting point for this problem.Lenet has shown its dominance in classification problems thats why I thought of choosing the same architecture.
    
    
* What were some problems with the initial architecture?

    The training and validation results for initial architecure was low and reason may be because of the less no of convolutional layers(as compared to other architecures) and 3 hidden dense connected layers.
    
    
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

    I decided to add one more convolutional layer so that the data is mapped to a higher dimension than 16 so i used 416 filters in the convolutional layer and since the output is 1*1*416 ,there wont be much effect to the bflops.
    

* Which parameters were tuned? How were they adjusted and why?

    Dropout keep probability ,epoch no and learning rate are adjusted since these can affect the training.for eg: esp high value of learning rate can cause training to take large steps and miss the optimum point.
    
    
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

    Convolution layers can extract more features than dense connected network since in convolution,neighbourhood pixels are also taken into account according to filter weights so it can take spatial distribution of data which is important for pictures.Also DNN requires flattened data and for images,it will be large size tensor.
    Dropout layers can be used for regularization since fullyconnected layers are sensitive to overfitting so by using dropout layers,we can 'drop' some of the weights.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

##### Test image1
![alt text][image4]
Compared to other signs,this sign is difficult to identify since brightness is less and gray scale image can make the model more confused ie missrate can be higher.

##### Test image2
![alt text][image5]
for this image color contrast is more and is difficult for the classifier to identify it from background.

##### Test image3
![alt text][image6]
Out of other images,this can be easily classified since features can be easily extracted and is highly aligned with training data.

##### Test image4
![alt text][image7]
I think this image is also fairly simple to classify as features are not blocked by any dust or oblique particles.

##### Test image5
![alt text][image8]
Compared to other images,this image is blurred one and due to high contrast colors model may fail in detecting this unless a similar image present in training set

##### Test image6
![alt text][image9]
This image is very similar to one of the images in training set except this image is brighter and is having more clarity.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Rightofway  			| Rightofway 									|
| Noentry				| Noentry										|
| RightOnly	      		| RightOnly 					 				|
| RoadWork  			| RoadWork          							|
| PriorityRoad 			| PriorityRoad         							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
![alt text][image10]
![alt text][image11]




