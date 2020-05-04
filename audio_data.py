# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:51:38 2020

@author: HP
"""

import os

def data(train_dir):
    audio_files = os.listdir(train_dir)
    aud_a=[]
    aud_b=[]
    aud_y=[]
    for i in range(len(audio_files)):
        a_path=audio_files[i]
            for j in range(len(audio_files)):
                b_path=audio_files[j]
                aud_a.append(spectrogram(train_dir + '\\'+a_path))
                aud_b.append(spectrogram(train_dir + '\\'+b_path))
                if(a_path[0]==b_path[0]):
                    aud_y.append(ord(a_path[0])-48)
                else:
                    aud_y.append(-1)
    return aud_a,aud_b,aud_y