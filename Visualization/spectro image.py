import math
import subprocess
from os import walk
import csv
import shutil
import numpy as np
import cv2
import pandas as pd
from scipy.io import wavfile
import librosa
import math
from scipy import signal
from scipy.io import wavfile
import librosa
from librosa import display
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal


#move to current working directory
os.chdir("Downloads\\Process Data")

#set the destination and source path
dir = <set the path to the source directory : #train #valid #test>
image = <folder path to save the images>

#define soundwave_to_np_spectogram
def get_short_time_fourier_transform(soundwave):
    return librosa.stft(soundwave, n_fft=256)

def short_time_fourier_transform_amplitude_to_db(stft):
    return librosa.amplitude_to_db(stft)

def soundwave_to_np_spectogram(soundwave):
    step1 = get_short_time_fourier_transform(soundwave)
    step2 = short_time_fourier_transform_amplitude_to_db(step1)
    step3 = step2/100
    return step3

#use audio and create spectogram and save it
for audio_number in os.listdir(dir):
    audio, sr = librosa.load(os.path.join(dir, audio_number))
    spectrogram = librosa.stft(audio)
    plt.figure()
    spec = soundwave_to_np_spectogram(audio)
    librosa.display.specshow(spec, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.savefig(os.path.join(image_test_d, "{}.png".format(audio_number)), bbox_inches='tight')
