Python 3.10.6 (tags/v3.10.6:9c7b4bd, Aug  1 2022, 21:53:49) [MSC v.1932 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import os
>>> import librosa
>>> import matplotlib.pyplot as plt
>>> import wave
>>> import scipy
>>> from scipy.io import wavfile
>>> import noisereduce as nr
>>> import subprocess
>>> import glob
>>> from glob import glob
>>> os.getcwd()
'C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python310'
>>> os.chdir('Downloads\DAIC Project')
>>> os.chdir('DAIC Dataset')
>>> os.getcwd()
'C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python310\\Downloads\\DAIC Project\\DAIC Dataset'
>>> os.chdir('audio')
>>> os.listdir()
['300_AUDIO.wav', '301_AUDIO.wav', '302_AUDIO.wav', '303_AUDIO.wav', '304_AUDIO.wav', '305_AUDIO.wav', '306_AUDIO.wav', '307_AUDIO.wav', '308_AUDIO.wav', '309_AUDIO.wav', '310_AUDIO.wav', '311_AUDIO.wav', '312_AUDIO.wav', '313_AUDIO.wav', '314_AUDIO.wav', '315_AUDIO.wav', '316_AUDIO.wav', '317_AUDIO.wav', '318_AUDIO.wav', '319_AUDIO.wav', '320_AUDIO.wav', '321_AUDIO.wav', '322_AUDIO.wav', '323_AUDIO.wav', '324_AUDIO.wav', '325_AUDIO.wav', '326_AUDIO.wav', '327_AUDIO.wav', '329_AUDIO.wav', '330_AUDIO.wav', '331_AUDIO.wav', '332_AUDIO.wav', '333_AUDIO.wav', '334_AUDIO.wav', '335_AUDIO.wav', '336_AUDIO.wav', '337_AUDIO.wav', '338_AUDIO.wav', '339_AUDIO.wav', '340_AUDIO.wav', '341_AUDIO.wav', '344_AUDIO.wav', '345_AUDIO.wav', '346_AUDIO.wav', '347_AUDIO.wav', '349_AUDIO.wav', '350_AUDIO.wav', '351_AUDIO.wav', '352_AUDIO.wav', '353_AUDIO.wav', '355_AUDIO.wav', '356_AUDIO.wav', '357_AUDIO.wav', '358_AUDIO.wav', '359_AUDIO.wav', '360_AUDIO.wav', '361_AUDIO.wav', '362_AUDIO.wav', '364_AUDIO.wav', '365_AUDIO.wav', '366_AUDIO.wav', '367_AUDIO.wav', '368_AUDIO.wav', '370_AUDIO.wav', '371_AUDIO.wav', '372_AUDIO.wav', '373_AUDIO.wav', '374_AUDIO.wav', '375_AUDIO.wav', '376_AUDIO.wav', '377_AUDIO.wav', '378_AUDIO.wav', '379_AUDIO.wav', '380_AUDIO.wav', '381_AUDIO.wav', '382_AUDIO.wav', '383_AUDIO.wav', '384_AUDIO.wav', '385_AUDIO.wav', '386_AUDIO.wav', '387_AUDIO.wav', '388_AUDIO.wav', '389_AUDIO.wav', '390_AUDIO.wav', '391_AUDIO.wav', '392_AUDIO.wav', '393_AUDIO.wav', '395_AUDIO.wav', '396_AUDIO.wav', '397_AUDIO.wav', '399_AUDIO.wav', '400_AUDIO.wav', '401_AUDIO.wav', '402_AUDIO.wav', '403_AUDIO.wav', '404_AUDIO.wav', '405_AUDIO.wav', '406_AUDIO.wav', '407_AUDIO.wav', '408_AUDIO.wav', '409_AUDIO.wav', '410_AUDIO.wav', '411_AUDIO.wav', '412_AUDIO.wav', '413_AUDIO.wav', '414_AUDIO.wav', '415_AUDIO.wav', '416_AUDIO.wav', '417_AUDIO.wav', '418_AUDIO.wav', '419_AUDIO.wav', '420_AUDIO.wav', '421_AUDIO.wav', '422_AUDIO.wav', '423_AUDIO.wav', '424_AUDIO.wav', '425_AUDIO.wav', '426_AUDIO.wav', '427_AUDIO.wav', '428_AUDIO.wav', '429_AUDIO.wav', '430_AUDIO.wav', '431_AUDIO.wav', '432_AUDIO.wav', '433_AUDIO.wav', '434_AUDIO.wav', '435_AUDIO.wav', '436_AUDIO.wav', '437_AUDIO.wav', '438_AUDIO.wav', '439_AUDIO.wav', '440_AUDIO.wav', '441_AUDIO.wav', '442_AUDIO.wav', '443_AUDIO.wav', '444_AUDIO.wav', '445_AUDIO.wav', '446_AUDIO.wav', '447_AUDIO.wav', '448_AUDIO.wav', '449_AUDIO.wav', '450_AUDIO.wav', '451_AUDIO.wav', '452_AUDIO.wav', '453_AUDIO.wav', '454_AUDIO.wav', '455_AUDIO.wav', '456_AUDIO.wav', '457_AUDIO.wav', '458_AUDIO.wav', '459_AUDIO.wav', '461_AUDIO.wav', '462_AUDIO.wav', '463_AUDIO.wav', '464_AUDIO.wav', '465_AUDIO.wav', '466_AUDIO.wav', '467_AUDIO.wav', '468_AUDIO.wav', '469_AUDIO.wav', '470_AUDIO.wav', '471_AUDIO.wav', '472_AUDIO.wav', '473_AUDIO.wav', '474_AUDIO.wav', '475_AUDIO.wav', '476_AUDIO.wav', '477_AUDIO.wav', '478_AUDIO.wav', '479_AUDIO.wav', '480_AUDIO.wav', '481_AUDIO.wav', '482_AUDIO.wav', '483_AUDIO.wav', '484_AUDIO.wav', '485_AUDIO.wav', '486_AUDIO.wav', '487_AUDIO.wav', '488_AUDIO.wav', '489_AUDIO.wav', '490_AUDIO.wav', '491_AUDIO.wav', '492_AUDIO.wav']
>>> os.getcwd()
'C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python310\\Downloads\\DAIC Project\\DAIC Dataset\\audio'
>>> path = 'C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python310\\Downloads\\DAIC Project\\DAIC Dataset\\audio'
>>> audio = glob(path + '/*.wav')
>>> len(audio)
183
>>> for path, directories, files in os.walk(path):
...     for audio in files:
...             if audio.endswith(".wav"):
...                     rate, data = wavfile.read(audio)
...                     reduced_noise = nr.reduce_noise(y=data, sr=rate)
...                     cleaned = audio.replace(".wav", "_cleaned.wav")
...                     wavfile.write(cleaned, rate, reduced_noise)
...

