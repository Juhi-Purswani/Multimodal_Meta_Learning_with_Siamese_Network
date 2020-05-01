# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:10:53 2020

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

def image_feat_network(input_shape):
    seq = Sequential()
    seq.add(Convolution2D(32, 3, 3, input_shape=input_shape,
                          border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))  
    seq.add(Dropout(.25))
    seq.add(Convolution2D(64, 3, 3, border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th')) 
    seq.add(Dropout(.25))
    seq.add(Reshape((1,64, 60)))
    return seq

def audio_feat_network(input_shape):
    seq = Sequential()
    seq.add(Convolution2D(64, 39, 9, input_shape=input_shape,
                          border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(1, 3)))  
    seq.add(Dropout(.25))
    seq.add(Convolution2D(64, 1, 10, border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(1, 28), dim_ordering='th')) 
    seq.add(Dropout(.25))
    seq.add(Reshape((1,64, 329)))
    return seq