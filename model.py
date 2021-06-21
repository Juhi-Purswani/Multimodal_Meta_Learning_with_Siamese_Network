# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:15:25 2020

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
import feature_extractor
import fusion

def siamese_network(input_dim_img,input_dim_aud):
    
    img_a = Input(shape=input_dim_img)
    img_b = Input(shape=input_dim_img)
    aud_a = Input(shape=input_dim_aud)
    aud_b = Input(shape=input_dim_aud)

    img_network = feature_extractor.image_feat_network(input_dim_img)
    feat_img_a = img_network(img_a)
    feat_img_b = img_network(img_b)
    
    aud_network = feature_extractor.audio_feat_network(input_dim_aud)
    feat_aud_a = aud_network(aud_a)
    feat_aud_b = aud_network(aud_b)
    
    concat_a = Concatenate(axis=2)([feat_img_a, feat_aud_a])
    concat_b = Concatenate(axis=2)([feat_img_b, feat_aud_b])
    
    input_dim = (27,37,32)
    base_network = fusion.multi_modal_network(input_dim)
    feat_vecs_a = base_network(concat_a)
    feat_vecs_b = base_network(concat_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a ,feat_vecs_b])
    
    rms = RMSprop()
    model = Model(inputs=[img_a, aud_a, img_b,aud_b], outputs=distance)
    return rms,model

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()
