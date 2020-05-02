# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:17:29 2020

@author: HP
"""
import matplotlib.pyplot as plt
import librosa
import librosa.display

def spectrogram(file_name):
    x, sr = librosa.load(file_name, sr=44100)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    return Xdb