#import relevant libraries
import os
import subprocess
from os import walk
import csv
import pydub
from pydub import AudioSegment
import pandas as pd
##setting up working directory
os.getcwd()
os.chdir('<change the directory to where the audio file is stored>')
dir = <set the path to targeted folder/directory>

#split cleaned audios into multiple chunks of each, according to participant's start and end timing
for path, directories, files in os.walk(dir):
    k = 0
    for audio in files:
            if audio.endswith("_TRANSCRIPT.csv"):
                    df = pd.read_csv(os.path.join(path,audio), sep= '\t')
                    df = df[df.speaker == 'Participant']
                    k = k + 1
            if audio.endswith("_cleaned.wav"):
                    tempaudio = AudioSegment.from_wav(os.path.join(path,audio))
                    k = k + 1
            if k == 2:
                    for index, row in df.iterrows():
                            t1 = row['start_time'] * 1000
                            t2 = row['stop_time'] * 1000
                            splitaudio = tempaudio[t1:t2]
                            splitaudio.export(os.path.join(path, '{}_{}_SPLIT.wav'.format(t1,t2)), format="wav")


#merging all the splitted chunks and combine it as one audio and replace it as the final audio output(name)
for path, directories, files in os.walk(dir):
    k = 0
    for audio in files:
            if audio.endswith("_cleaned.wav"):
                    participant = audio.replace("_cleaned.wav", "_final.wav")
            if audio.endswith("_SPLIT.wav"):
                    audio = AudioSegment.from_wav(os.path.join(path,audio))
                    if k == 0:
                            combined = audio
                            k = 1
                    else:
                            combined = combined + audio
    combined.export(os.path.join(path,participant), format="wav")


#increase the volume (trial) by 10and save it as increased volume
from pydub.playback import play
wav_file = pydub.AudioSegment.from_file(dir)
new_wav_file = wav_file + 10
new_wav_file.export(os.path.join(path,'_increased_volume.wav'), format="wav")
