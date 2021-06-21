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
   seq.add(Convolution2D(32, kernel_size=(3,3), input_shape=(56,56,1)))
   seq.add(Activation('relu'))
   seq.add(MaxPooling2D(pool_size=(2, 2)))
   seq.add(Dropout(.25))
   return seq

def audio_feat_network(input_shape):
    seq = Sequential()
    seq.add(Convolution2D(32, kernel_size=(3,3), input_shape=(1025,47,1)))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(4, 2)))
    seq.add(Dropout(.25))
    seq.add(Convolution2D(32, kernel_size=(3,3)))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(4, 2)))
    seq.add(Convolution2D(32, kernel_size=(4,1)))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(1, 1)))
    seq.add(Convolution2D(32, kernel_size=(4,1)))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(1, 1)))
    seq.add(Convolution2D(32, kernel_size=(3,1)))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 1)))
    return seq
