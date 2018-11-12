import dill
import pyaudio
import wave
import pywt
import numpy as np
from os import system
import soundfile as sf
import matplotlib.pyplot as plt
# from sklearn.svm import SVC
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

    x=mags

    #ca: koef aproximasi, cd:koef.detail, haar:filter)
    c = pywt.wavedec(x, mode, level=level)

    print(c[0])
    print ("Plot Data\n")
	#plot original data
    plt.figure(1)
    plt.title('Original Signal Wave')
    plt.plot(data)

    # plt.figure(2)
    # plt.title('emphasized_signal')
    # plt.plot(emphasized_signal) 

    plt.figure(3)
    plt.title('hamming signal')
    plt.plot(mags)

	#plot wavelet data
    plt.figure(4)
    plt.title('wavelet Signal Wave')
    plt.plot(c[0])

   
    plt.show()

    return c[0]


wavelet = wavedec("temp.wav")


wavelet = np.expand_dims(wavelet, axis=0)

res = (classifier.predict(wavelet))


dist2hptr=classifier.decision_function(wavelet)
tr_y=classifier.predict(wavelet)

print ("RESULT : \n")

print ('Debug class : ', res)
print ('dist2hptr: ', dist2hptr)

if res == 1: 
    print ('Suara Terdaftar')
    system('python test_success_indicator.py') 
else:
    print ('Unknown')
    system('python test_fail_indicator.py')


