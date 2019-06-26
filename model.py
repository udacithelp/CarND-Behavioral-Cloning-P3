import csv
import os
# import cv2
from cv2 import cv2
import numpy as np
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
#
from keras.layers import Convolution2D
# from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
#
from keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(7)

lines = []
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# with open(csv_file, 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         steering_center = float(row[3])
#
#         # create adjusted steering measurements for the side camera images
#         correction = 0.2 # this is a parameter to tune
#         steering_left = steering_center + correction
#         steering_right = steering_center - correction
#
#         # read in images from center, left and right cameras
#         path = "..." # fill in the path to your training IMG directory
#         img_center = process_image(np.asarray(Image.open(path + row[0])))
#         img_left = process_image(np.asarray(Image.open(path + row[1])))
#         img_right = process_image(np.asarray(Image.open(path + row[2])))
#
#         # add images and angles to data set
#         car_images.extend(img_center, img_left, img_right)
#         steering_angles.extend(steering_center, steering_left, steering_right)
#
#
# Figuring out how much to add or subtract from the center angle will involve some experimentation.
#
# During prediction (i.e. "autonomous mode"), you only need to predict with the center camera image.
#
# It is not necessary to use the left and right images to derive a successful model. Recording recovery driving from the sides of the road is also effective.
        

images = []
measurements = []
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split("/")[-1]
    current_path = os.path.join(os.getcwd(), "data/IMG/", filename)
    # print(os.getcwd())
    #     NOTE: cv2.imread will get images in BGR format, while drive.py uses RGB. In the video above one way you could keep the same image formatting is to do "image = ndimage.imread(current_path)" with "from scipy import ndimage" instead.
    bgr = cv2.imread(current_path)
    rgb = bgr[...,::-1]
    image = rgb
    # image = ndimage.imread(current_path)
    # image = cv2.imread(current_path)
    # print(current_path)
    # bgr = cv2.imread(current_path)
    # image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# print(measurements)

X_train = np.array(images)
y_train = np.array(measurements)

def dumbmodel():
    model = Sequential()
    #
    IM_DIM=(160,320,3)
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=IM_DIM))
    # model.add(Flatten(input_shape=IM_DIM))
    model.add(Flatten())
    model.add(Dense(1))
    return model



def lenet():
    model = Sequential()    
    IM_DIM=(160,320,3)
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=IM_DIM))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def myNet():
    model = Sequential()    
    IM_DIM=(160,320,3)
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=IM_DIM))
    model.add(Lambda(lambda x: (x/255.0) - 0.5))
#    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=IM_DIM))
#    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    #
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(layers.Dropout(0.5))  # rate: float between 0 and 1. Fraction of the input units to drop.
    #
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nvidia():
    model = Sequential()    
    IM_DIM=(160,320,3)
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=IM_DIM))
    model.add(Lambda(lambda x: (x/255.0) - 0.5))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(layers.Dropout(0.1))  # rate: float between 0 and 1. Fraction of the input units to drop.
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model    


# model = myNet()
model = nvidia()

# DATA AUGMENTATION # MEH
# image_flipped = np.fliplr(image)
# measurement_flipped = -measurement
    
model.compile(loss="mse", optimizer="adam")
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)
# maybe early stopping?
# https://chrisalbon.com/deep_learning/keras/neural_network_early_stopping/
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=20, callbacks=callbacks)

model.save('model.h5')


# history_object = model.fit_generator(train_generator, samples_per_epoch =
#     len(train_samples), validation_data = 
#     validation_generator,
#     nb_val_samples = len(validation_samples), 
#     nb_epoch=5, verbose=1)


# ### print the keys contained in the history object
# print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

