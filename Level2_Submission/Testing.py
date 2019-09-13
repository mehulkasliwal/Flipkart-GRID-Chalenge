#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


import cv2
import pandas as pd
import glob
import keras
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense,GlobalAveragePooling2D,Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras import optimizers


# # Loading the npy file of test images

# In[ ]:


X_test = np.load('./Test_Processed_192.npy')


# # Loading the npy files of mean and std obtained in from Training Dataset

# In[ ]:


mean = np.load('mean.npy')
std = np.load('std.npy')


# # Scaling the test images

# In[ ]:


X_test = X_test/255
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_test = X_test - mean
X_test = X_test / std
print(X_test.shape)


# # Model Architecture

# In[ ]:


input_1 = Input(shape = (192,192,1))
# Block 1
x = layers.Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv1')(input_1)
x = layers.Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = layers.Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv1')(x)

x = layers.Conv2D(128, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block2_conv2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

x = layers.Dropout(0.2)(x)

# Block 3
x = layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv1')(x)

x = layers.Conv2D(256, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block3_conv2')(x)

x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
x = layers.Dropout(0.2)(x)

# Block 4
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv1')(x)

x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block4_conv2')(x)

x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
x = layers.Dropout(0.2)(x)
# Block 5
x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv1')(x)

x = layers.Conv2D(512, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block5_conv2')(x)

x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4,kernel_initializer='normal')(x)
model = Model(inputs=input_1, outputs=predictions)


# # Compiling the Model
# 

# In[ ]:


model.summary()
#rms = optimizers.RMSprop(lr=0.0001, rho=0.9)
adm = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adm,metrics=['accuracy'])


# # Loading the weights obtained after training the model

# In[ ]:


model.load_weights('Best_Model10.h5')


# # Prediction

# In[ ]:


Y_predicted = model.predict(X_test)
print(Y_predicted.shape)
for i in range(0,Y_predicted.shape[0]):
    Y_predicted[i][0] = Y_predicted[i][0] / xscale
    Y_predicted[i][1] = Y_predicted[i][1] / xscale
    Y_predicted[i][2] = Y_predicted[i][2] / yscale
    Y_predicted[i][3] = Y_predicted[i][3] / yscale


# # Exporting the predictions to .csv file

# In[ ]:


X_label = np.load('./X_test.npy')
X_label = X_label.reshape(X_label.shape[0],1)
print(X_label.shape)
for i in range(0,X_label.shape[0]):
  X_label[i] = X_label[i][0].split('/')[2]

print(X_label.shape)
print(X_label)
DF = np.concatenate((X_label, Y_predicted), axis=1)
print(DF.shape)
print(DF)
type(DF[0][2])

out = pd.DataFrame({'image_name':DF[:,0],'x1':DF[:,1],'x2':DF[:,2],'y1':DF[:,3],'y2':DF[:,4]})
out.to_csv('test.csv')

