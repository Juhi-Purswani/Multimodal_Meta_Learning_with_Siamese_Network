# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:12:15 2020

@author: HP
"""
import image_data
import audio_data

def data_generate(train_dir):
    img_a,img_b,img_y=image_data.data()
    aud_a,aud_b,aud_y=audio_data.data(train_dir)
    label=[]
    for i in range(len(img_a)):
        if(img_y==-1 or aud_y == -1):
            label.append(0)
        elif(img_y==aud_y):
            label.append(1)
        else:
            label.append(0)
    return img_a,aud_a,img_b,aud_b,label