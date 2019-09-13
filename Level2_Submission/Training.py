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


# # Extracting useful images from Dataset

# In[2]:


files = glob.glob ("./images/*.png")
files = np.asarray(files)
data_train = pd.read_csv("training.csv")
data_test = pd.read_csv("test.csv")
Y_training = data_train.iloc[:,1:]
X_training = data_train.iloc[:,0:1]
Y_testing = data_test.iloc[:,1:]
X_testing = data_test.iloc[:,0:1]
collectionArr = files
X_train = []
X_test = []
Y_train =[]
Y_test = []
counter = 0
for each in collectionArr:
    image_name = each.split('/')[2]
    indexer1 = X_training.index[X_training['image_name'] == image_name]
    indexer2 = X_testing.index[X_testing['image_name']== image_name]
    if(indexer1.empty):
        pass
    else:
        X_train.append(each)
        Y_train.append(Y_training.iloc[indexer1,:].values)
    if(indexer2.empty):
        pass
    else:
        X_test.append(each)
    counter+=1
    print(counter)
np.save('X_train',X_train)
np.save('Y_train',Y_train)
np.save('X_test',X_test


# # Preprocessing the Images

# In[ ]:


files = np.load('X_train.npy')
X64 = []
for myFile in files:
    image = cv2.imread (myFile)
    image = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
    resized64 = cv2.resize(image,(192,192))
    X64.append (resized64)
    
X64 = np.asarray(X64)
print(X64.shape)
np.save("Train_Processed_192",X64)

files = np.load('X_test.npy')
X64_test = []
for myFile in files:
    image = cv2.imread (myFile)
    image = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
    resized64 = cv2.resize(image,(192,192))
    X64_test.append (resized64)
    
X64_test = np.asarray(X64_test)
print(X64_test.shape)
np.save("Train_Processed_192",X64_test)


# # Importing and scaling the training images

# In[3]:


xscale = 192/640
yscale = 192/480
X_train = np.load('./Train_Processed_192.npy')
Y_train = np.load('./Y_train.npy')
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
print(X_train.shape)
print(Y_train.shape)
Y_train_new = []
for i in range(0,Y_train.shape[0]):
    Y_train_new.append(Y_train[i][0])
Y_train_new = np.asarray(Y_train_new)
print(Y_train_new.shape)
print(Y_train_new.shape[0])
for i in range(0,Y_train_new.shape[0]):
    Y_train_new[i][0] = Y_train_new[i][0] * xscale
    Y_train_new[i][1] = Y_train_new[i][1] * xscale
    Y_train_new[i][2] = Y_train_new[i][2] * yscale
    Y_train_new[i][3] = Y_train_new[i][3] * yscale
X_train = X_train/255
mean = np.mean(X_train,axis=0)
std = np.std(X_train,axis=0)
print("Before ",mean.shape,std.shape)
X_train = X_train - mean
X_train = X_train / std
print("After ",X_train.mean(),X_train.std())
np.save("mean.npy",mean)
np.save("std.npy",std)


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

# In[ ]:


model.summary()
#rms = optimizers.RMSprop(lr=0.0001, rho=0.9)
adm = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adm,metrics=['accuracy'])


# # Training the Model

# In[ ]:


filepath="weights-improvement-{epoch:02d}-{acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True,period=1)
earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=3,verbose=1, mode='auto',restore_best_weights=True)
callbacks_list = [earlystop,checkpoint]
history=model.fit(X_train,Y_train_new, batch_size = 8, callbacks = callbacks_list,epochs =35,verbose = 1)


# # Saving the weights of the Trained Model

# In[ ]:


model.save_weights('Best_Model10.h5')

