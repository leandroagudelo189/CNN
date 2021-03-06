#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:43:50 2017

@author: Leo
"""

### Convolutional Neural Networks
# Computer recognition of categorical data

### 1. Build the CNN

# importing libraries and packages

from keras.models import Sequential # to initialize the NN as a sequence of layers
from keras.layers import Convolution2D # for step 1 in convolutional layers
from keras.layers import MaxPooling2D 
from keras.layers import Flatten # convert the pooled features into one single vectors
from keras.layers import Dense # to add the fully connected layers

# start the CNN
model = Sequential() # create an object of our CNN

# Add the convolutional layers 
model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))   # we will apply a method on this object (filters = feature_maps with #of rows and columns)

# Reduce the filters by Max pooling
model.add(MaxPooling2D(pool_size=(2,2)))

# add an additional or second convolutional layer after the max pooling
model.add(Convolution2D(32, (3, 3),  activation='relu')) # we don't need to include the input_shape since we have the pooled features (keras will notice it)
model.add(MaxPooling2D(pool_size=(2,2)))

# Flattening= take our pooled maps and convert them into a vector
model.add(Flatten())

# make a classic ANN "fully connected"
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# compile the CNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# fit the CNN to the images
# use image augmentation to preprocess images to avoid overfitting
# use flow_from_directory if you have your dataset organized in folders
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64), # to get better accuracy one can increase the size here (more pixels)
                                                batch_size=32,
                                                class_mode='binary')


test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
        

model.fit_generator( training_set,
                    steps_per_epoch=8000,  #total samples
                    epochs=2,
                    validation_data=test_set,
                    validation_steps=2000)


# to improve the accuracy of the model we can do different things
# 1. Add an additional convolutional layer
# 2. Tune the ANN. Hyperparameter tuning or add an additional fully-connected layer


### now predict your own pictures

import numpy as np
from keras.preprocessing import image
test_new_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
# now we need to add a new dimension to the image to match the input shape of our training dataset
test_new_image = image.img_to_array(test_new_image)
# we need to add an additional dimension to our 3D-array     
# because the conv2d method only accepts input in a batch (index) therefore 4 dimensions
test_new_image = np.expand_dims(test_new_image, axis=0)

prediction = model.predict(test_new_image)
training_set.class_indices

if prediction[0][0] == 1:
    pred_final = 'dog'
else:
    pred_final = 'cat'
    
    




