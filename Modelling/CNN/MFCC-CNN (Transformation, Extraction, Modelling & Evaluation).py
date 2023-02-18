#import relevant libraries
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
from librosa import display
import pandas as pd
from sklearn.metrics import confusion_matrix
import keras
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Embedding
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, LSTM, Bidirectional
import tensorflow as tf
import numpy as np
import itertools
import math
import time
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, LSTM, Bidirectional
import os
from pydub import AudioSegment
from scipy import signal
import keras
from keras.models import Sequential
from keras.layers import Dense
import os
import re
import numpy as np
import pandas as pd
import scipy.io.wavfile as sw
import python_speech_features as psf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import signal
import spafe
from spafe.features.gfcc import gfcc
import numpy as np
from scipy.io.wavfile import read as wavread
import scipy
from spafe.features.gfcc import gfcc


#set path for the transcript file for train and test set
os.chdir("Downloads\\Model\\csv")
train_df = pd.read_csv('train_split1.csv')
train_labels = list(train_df['PHQ8_Binary'])
os.listdir()
['test_split.csv', 'train_split.csv', 'valid_split.csv']
valid_df = pd.read_csv('valid_split.csv')
valid_labels = list(valid_df['PHQ8_Binary'])


#data augmentation PHQ-8 score (predicted value)
i = 0
j = 0
k = 0
number_list_train = ['']
number_list_test = ['']
number_list_dev = ['']
file_count_train = 780
file_count_test = 260
file_count_dev = 240
input_array_train = ['']
phq8_array_train = ['']
input_array_train *= file_count_train
phq8_array_train *= file_count_train
number_list_train *= file_count_train
input_array_test = ['']
phq8_array_test = ['']
input_array_test *= file_count_test
phq8_array_test *= file_count_test
number_list_test *= file_count_test
input_array_dev = ['']
phq8_array_dev = ['']
input_array_dev *= file_count_dev
phq8_array_dev *= file_count_dev
number_list_dev *= file_count_dev

#data augmentation PHQ-8 score (predicted value)
phq8_array_train = train_labels
phq8_array_dev = valid_labels


train_split = '<move to the train set>'
dev_split = '<move to the test set>'

S_mfcc_train = ['']*(file_count_train)
S_mfcc_train_normalized = ['']*(file_count_train)
i = 0
for file in os.listdir(train_split):
    if file.endswith(".wav"):
        soundwave_path = os.path.join(train_split, file)
        X, sr = librosa.load(soundwave_path)
        mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=13)
        S_mfcc_train[i] = mfccs	
        S_mfcc_train_normalized[i] = tf.math.log(tf.abs(S_mfcc_train[i]) + 0.01)
        i += 1



S_mfcc_dev = ['']*(file_count_dev)
S_mfcc_dev_normalized = ['']*(file_count_dev)
i = 0
for file in os.listdir(dev_split):
    if file.endswith(".wav"):
        soundwave_path = os.path.join(dev_split, file)
        X, sr = librosa.load(soundwave_path)
        mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=13)
        S_mfcc_dev[i] = mfccs	
        S_mfcc_dev_normalized[i] = tf.math.log(tf.abs(S_mfcc_dev[i]) + 0.01)
        i += 1


# Size of the spectogram. Every sample has the same lenght so only 1 size is calculated.
alto, ancho = np.shape(S_mfcc_train_normalized[0])
alto, ancho = np.shape(S_mfcc_dev_normalized[0])


# Reshaping for a convolutional network
S_mfcc_train_normalized_array = np.ndarray(shape = (780,13,ancho))
S_mfcc_dev_normalized_array = np.ndarray(shape = (240,13,ancho))

i = 0
for train in S_mfcc_train_normalized:
    S_mfcc_train_normalized_array[i] = train[0:13,:]
    i += 1


i = 0
for train in S_mfcc_dev_normalized:
    S_mfcc_dev_normalized_array[i] = train[0:13,:]
    i += 1


S_mfcc_train_normalized_array = S_mfcc_train_normalized_array.reshape(780,13,ancho,1)
S_mfcc_dev_normalized_array = S_mfcc_dev_normalized_array.reshape(240,13,ancho,1)


phq8_array_train_extended = np.ndarray(shape = (780,1))
phq8_array_dev_extended = np.ndarray(shape = (240,1))

j = 0
for valor in phq8_array_train:
     phq8_array_train_extended[j] = valor
     j += 1



j = 0
for valor in phq8_array_dev:
     phq8_array_dev_extended[j] = valor
     j += 1

#Creation of Convolutional model
num_classes = 2
batch_size = 64
epochs = 10

# convert class vectors to binary class matrices
phq8_array_train = keras.utils.to_categorical(phq8_array_train_extended, num_classes)
phq8_array_dev = keras.utils.to_categorical(phq8_array_dev_extended, num_classes)


#CNN
num_classes = 2
batch_size = 10
epochs = 100

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='valid', strides=1, input_shape=(13, ancho, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))
model.add(Conv2D(32, (1, 3), padding='valid', strides=1,input_shape=(13,ancho,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))#cancelled
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(S_mfcc_train_normalized_array, phq8_array_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(S_mfcc_dev_normalized_array, phq8_array_dev))


score_train = model.evaluate(S_mfcc_train_normalized_array, phq8_array_train, verbose=0)
print('Train accuracy:', score_train[1])
score_test = model.evaluate(S_mfcc_dev_normalized_array, phq8_array_dev, verbose=0)
print('Test accuracy:', score_test[1])
print('Test loss:', score_train[0])
print('Test loss:', score_test[0])


#EVALUATION METRICS

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

S_mfcc_dev_normalized_array.shape #4D array


y_true = phq8_array_dev[:, 0]
y_pred = model.predict(S_mfcc_dev_normalized_array)
y_pred_class = np.round(y_pred[:, 0])
conf_matrix = confusion_matrix(y_true, y_pred_class)
print("Confusion Matrix: \n", conf_matrix)

precision = precision_score(y_true, y_pred_class)
recall = recall_score(y_true, y_pred_class)
f1 = f1_score(y_true, y_pred_class)
accuracy = accuracy_score(y_true, y_pred_class)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)
print("Accuracy: ", accuracy)



# calculate sensitivity
sensitivity = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
print("Sensitivity: ", sensitivity)

# calculate specificity
specificity = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
print("Specificity: ", specificity)


#ROC CURVE GRAPH
from sklearn.metrics import roc_curve, roc_auc_score

y_pred_scores = model.predict(S_mfcc_dev_normalized_array)[:,0]
fpr, tpr, thresholds = roc_curve(y_true, y_pred_scores)
roc_auc = roc_auc_score(y_true, y_pred_scores)

plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
