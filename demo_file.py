import dill
import pyaudio
import wave
import pywt
import numpy as np
from os import system
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.signal import hann
from scipy.fftpack import rfft
# import csv
# import pandas as pd


print ("Loading svm-model.dill....")
# Load PNN

# df = pd.read_csv('wavelet.csv')

with open('svm-model.dill', 'rb') as f:
    classifier = dill.load(f)

def wavedec(audio, mode='haar', level=5):

    # read audio wav
    data, samplerate = sf.read(audio)  

     #cetak ukuran array data
    print(str(data.shape))

    #cetak samplerate
    print(str(samplerate))

    audio = data
    # apply a Hanning window
    window = hann(31744)
    audio = audio[0:31744] * window
    # fft
    mags = abs(rfft(audio))

    x = mags
    #ca: koef aproximasi, cd:koef.detail, haar:filter)
    c = pywt.wavedec(x, mode, level=level)

    print(c[0])

    return c[0]


wavelet = wavedec("temp.wav")

wavelet = np.expand_dims(wavelet, axis=0)
res = (classifier.predict(wavelet))
prob = (classifier.predict_proba(wavelet)[0])
prob_total = sum(prob)*100
prob_per_class_dictionary = dict(zip(classifier.classes_, prob))
winner = np.argmax(prob)


print ("probabilitas : \n", prob)
print ("probabilitas per class: \n", prob_per_class_dictionary)
print ("Predicted class : \n", res)

print (np.argmax(prob))

print ("RESULT : \n")
if prob[winner] > 0.62: 
    print ('Suara Terdaftar')
    system('python test_success_indicator.py') 
else:
    print ('Unknown')
    system('python test_fail_indicator.py')



