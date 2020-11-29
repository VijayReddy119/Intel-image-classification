# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 06:08:14 2020

@author: Vijay
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_sze = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('seg_train/seg_train',
                                                 target_size = (150, 150),
                                                 batch_size = batch_sze,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('seg_test/seg_test',
                                            target_size = (150, 150),
                                            batch_size = batch_sze,
                                            class_mode = 'categorical')

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(training_set,
                         steps_per_epoch = 14034/128,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 3000/128)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('seg_pred/seg_pred/19629.jpg', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices

#Saving Model
model.save_weights('CNN_Model_weights(intel-classification).h5')
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())
    
#Loading Model
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('model_weights.h5')