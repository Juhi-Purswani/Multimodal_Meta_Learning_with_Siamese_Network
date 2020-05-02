# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:08:47 2020

@author: HP
"""

import tensorflow as tf
import matplotlib.pyplot as plt

def data:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    count=[]
    for i in range(0,10):
        count.append(0)
    x_num=[]
    y_num=[]
    num=0
    i=0
    while(num!=10 and i<1000):
        if(count[y_train[i]]<2):
            x_num.append(x_train[i])
            y_num.append(y_train[i])
            count[y_train[i]] += 1
        elif(count[y_train[i]]==2):
            num += 1
            count[y_train[i]] = 3
        i = i+1
    img_a=[]
    img_b=[]
    img_y=[]
    for i in range(0,10):
        a = x_num[i]
        for j in range(0,10):
            img_a.append(a)
            img_b.append(x_num[j])
            if(y_num[i]==y_num[j]):
                img_y.append(y_num[i])
            else:
                img_y.append(-1)
    return img_a,img_b,img_y