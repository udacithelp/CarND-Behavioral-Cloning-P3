# **Behavioral Cloning** 

## Writeup Template

This is the completed writeup.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

While I tried several architectures discussed in lecture, I found a variation on the nvidia was most useful.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is the implementation of the nvidia network discussed in lecture:

- Convolutions, size 24, 36, 48, 64, 64 with filters 5x5 then 3x3
- Flattening layer
- Fully connected layers size 100, 50, 10, 1

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The following were undertaken:

  - The model contains a dropout layer with a small 0.1 dropout rate in order to reduce overfitting.
  - It also crops out extraneous regions of the image in order to prevent overfitting on non-essential portions of
the image, in things like the sky.
  - I used early stopping with a patience=4 parameter that I learned about googling here: https://chrisalbon.com/deep_learning/keras/neural_network_early_stopping/

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.  I just used the center camera and the
default data was sufficient.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started out with the simplest regression model that would get the car to go, then implemented the LeNet and
nvidia models discussed in class.  I changed cropping and early stopping parameters, but found the most
benefit from adding data augmentation and modifying dropout rates.

After I had something that worked, I gradually added code complexity in the form of the model_fit_generator.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network discussed above that was essentially the nvidia but with a dropout layer between the convolutional and flattening layer.

#### 3. Creation of the Training Set & Training Process

I had a lot of trouble getting the simulation to run in the first place, so I started my assignment only
with the training data given as a sample.

My preprocessing was very helpful, and consisted of flipping the images and measurements to double the training
data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as a start, but early stopping helped reduce that.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
