#import relevant libraries
import os
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
import spafe
from spafe.features.gfcc import gfcc

#move to current working directory
os.chdir("Downloads\\Process Data")

#set the destination and source path
dir = <set the path to the source directory : #train #valid #test>
image = <folder path to save the images>

#plotting
for audio_number in os.listdir(dir):
    # Load the audio file
    fs, sig = scipy.io.wavfile.read(os.path.join(train_path_d, audio_number))
    # Extract GFCC features
    gfccs = gfcc(sig, fs=fs, num_ceps=13)
    # Plot and save the GFCCs
    plt.figure()
    librosa.display.specshow(gfccs, x_axis='time')
    plt.colorbar()
    plt.title('GFCCs')
    plt.tight_layout()
    plt.savefig(os.path.join(image, "{}.png".format(audio_number)), bbox_inches='tight')
