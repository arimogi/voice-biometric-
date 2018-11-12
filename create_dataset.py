import os
import sys

import soundfile as sf #import audio wav
import numpy as np
import pywt
import h5py
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.io.wavfile import read
from scipy.signal import hann
from scipy.fftpack import rfft
import matplotlib.pyplot as plt


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

    c = pywt.wavedec(x, mode, level=level) 

    print(c[0])

    return c[0]

def wavelet_data(path):
    pathData = path

    # Cari data di semua folder
    dirs = os.listdir(pathData)
    print('Ditemukan %d folder' % (len(dirs)))

    # Cek dan buat folder hasil wavelet
    wav_dir = os.path.join(pathData, 'wavelet')
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)

    audio_data = []
    audio_label = []

    # Cek per manusia
    for dir in dirs:
        current_dir = os.path.join(pathData, dir)
        files = os.listdir(current_dir)
        print('Ditemukan %d files dalam folder %s' % (len(files), dir))

        # Setiap file suara
        total = 0
        for i, file in enumerate(files):
            # Ekstraksi fitur dengan wavelet
            wav = wavedec(os.path.join(current_dir, file), 'haar')

            audio_data.append(wav)
            audio_label.append(dir)

            total += 1

        print('audio diproses sebanyak %d' % (total))

    #print(audio_data) 

    return audio_data, audio_label

if __name__ == '__main__':
    wavelet_data(sys.argv[1])
