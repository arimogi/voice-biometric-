import sys 
from create_dataset import wavelet_data
import pywt #untuk wavelet (inklut normalisasi)
import dill #import untuk nimpen hasil pnn
from sklearn.preprocessing import LabelEncoder #label jdi integer
import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import normalize
# from sklearn.svm import SVC
from sklearn import svm
import time
import numpy as np
# import pandas as pd
# import csv

start_time = time.time()


def train(path="data"):
    audio_data, audio_label = wavelet_data(path)

    X = np.float32(audio_data) 
    print('shape X:', str(X.shape)) #untuk nampilin array x

    Y = audio_label
    print('shape Y: ', str(len(Y))) #untuk nampilin array y

    C = 1.0 
    classifier = svm.SVC(kernel='linear',C=C, class_weight='balanced',probability=True)
    classifier.fit(X , Y)
    with open('svm-model.dill', 'wb') as f: #hasil sebuah file model dngn nama pnn-mdel.dill
        dill.dump(classifier, f)

  
if __name__ == '__main__':
    train()
