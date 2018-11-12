import sys 
from create_dataset import wavelet_data
import pywt 
import dill
from sklearn.preprocessing import LabelEncoder 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import time
import numpy as np
# import csv
# import pandas as pd

start_time = time.time()


def train(path="data"):
    audio_data, audio_label = wavelet_data(path)

    X = np.float32(audio_data) 

    Y = audio_label

    classifier = svm.OneClassSVM(gamma=0.5, kernel='rbf', nu=0.1)
    
    # classifier = svm.OneClassSVM(kernel='rbf', degree=3, gamma=1, coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=None)
    classifier.fit(X)

   
    with open('svm-model.dill', 'wb') as f: 
        dill.dump(classifier, f)

if __name__ == '__main__':
    train()
