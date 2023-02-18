#Import Libraries
import os
import librosa
import matplotlib.pyplot as plt
import wave
import pyAudioAnalysis
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS

#Get current working directory
os.getcwd()
#set the audio destination path
path='<path where the uncleaned audios stored>'
[Fs, x] = aIO.read_audio_file(path)
segments = aS.silence_removal(x,
                              Fs,
                              0.020,
                              0.020,
                              smooth_window=1.0,
                              weight=0.3,
                              plot=True)
