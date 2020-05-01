# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:14:56 2020

@author: HP
"""
import re
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten,Concatenate,Reshape
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import tensorflow as tf

def multi_modal_network(input_shape):
    nb_filter = [6, 12]
    kernel_size = 3
    seq = Sequential()
    seq.add(Convolution2D(nb_filter[0], kernel_size, kernel_size, input_shape=input_shape,
                          border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))  
    seq.add(Dropout(.25))
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq