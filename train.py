# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:25:00 2020

@author: HP
"""

import re
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

import features
import model

train_dir = "train"
test_dir = "test"
batch_size = 64
epochs = 30
input_dim_img = (56,56,1)
input_dim_aud = (1025, 47,1)

img_a,aud_a,img_b,aud_b,labels = features.data_generate(train_dir)
opt,model = model.siamese_model(input_img_dim,input_aud_dim)

model.compile(loss=model.contrastive_loss, optimizer=opt)
model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
model.fit([img_a,aud_a,img_b,aud_b], labels, validation_split=.25,batch_size=batch_size, verbose=2, nb_epoch=epochs, callbacks=[es])
