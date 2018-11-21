
# coding: utf-8

# In[14]:
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os
import tensorflow as tf
import multiprocessing as mp

np.random.seed(0)


# In[15]:

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

batch_size = 128
num_classes = 10
epochs = 200
# Converting to float to be able to perform divison
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

mean_train = np.mean(x_train,axis=0)
mean_test = np.mean(x_test,axis=0)

x_train -= mean_train
x_test -= mean_test

x_train  /= np.std(x_train,axis=0)
x_test /= np.std(x_test,axis=0)

datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.16,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.16,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False
          ) # randomly flip images
#flatten 


# In[22]:

model = Sequential()

model.add(Conv2D(180, (5, 5), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(150, (1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding = 'same'))
model.add(Dropout(0.5))


model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding = 'same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# In[23]:

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.00001)
nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004)

model.compile(loss='categorical_crossentropy', optimizer= nadam, metrics=['accuracy'])

#now = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
#tensorboard = keras.callbacks.TensorBoard(log_dir='/home/mohamedyasser96/downloads/logs/' + now, histogram_freq=0, write_graph=True, write_images=True)



# In[24]:

datagen.fit(x_train[:40000]) #training 40000 and validating 10K

checkpointer = ModelCheckpoint(filepath="./Assin3_Trial3.hdf5", verbose=1, save_best_only=True, monitor='val_acc')

model.fit_generator(datagen.flow(x_train[:40000], Y_train[:40000],batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=(x_train[40000:], Y_train[40000:]),
                    workers=8, 
                    callbacks=[checkpointer])


# In[25]:

model.summary()

# Calculating ACCR
score = model.evaluate(x_test, Y_test, verbose=1)
print('Test score:', score[0])
print('ACCR:', score[1])


# In[ ]:



