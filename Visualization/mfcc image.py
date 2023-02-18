>>> import os
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

#plotting
for audio_number in os.listdir(dir):
    # Load the audio file
    audio, sr = librosa.load(os.path.join(dir, audio_number))
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)
    # Plot the MFCCs
    plt.figure()
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')
    plt.tight_layout()
    plt.savefig(os.path.join(image, "{}.png".format(audio_number)), bbox_inches='tight')
