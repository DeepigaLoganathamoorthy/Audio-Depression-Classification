#import relevant libraries
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


train_split = '<move to the train set>'
dev_split = '<move to the test set>'

# Spectograms of each segmented audio
def get_short_time_fourier_transform(soundwave):
    return librosa.stft(soundwave, n_fft=256)

def short_time_fourier_transform_amplitude_to_db(stft):
    return librosa.amplitude_to_db(stft)

def soundwave_to_np_spectogram(soundwave):
    step1 = get_short_time_fourier_transform(soundwave)
    step2 = short_time_fourier_transform_amplitude_to_db(step1)
    step3 = step2/100
    return step3


S_spec_train = ['']*(file_count_train)
S_spec_train_normalized = ['']*(file_count_train)
i = 0
for file in os.listdir(train_split):
    if file.endswith(".wav"):
        soundwave_path = os.path.join(train_split, file)
        X, sr = librosa.load(soundwave_path)
        S_spec_train[i] = abs(soundwave_to_np_spectogram(X))
        S_spec_train_normalized[i] = tf.math.log(tf.abs(S_spec_train[i]) + 0.01)
        i += 1



S_spec_dev = ['']*(file_count_dev)
S_spec_dev_normalized = ['']*(file_count_dev)
i = 0
for file in os.listdir(dev_split):
    if file.endswith(".wav"):
        soundwave_path = os.path.join(dev_split, file)
        X, sr = librosa.load(soundwave_path)
        S_spec_dev[i] = abs(soundwave_to_np_spectogram(X))
        S_spec_dev_normalized[i] = tf.math.log(tf.abs(S_spec_dev[i]) + 0.01)
        i += 1




# Size of the spectogram. Every sample has the same lenght so only 1 size is calculated.
alto, ancho = np.shape(S_spec_train_normalized[0])
alto, ancho = np.shape(S_spec_dev_normalized[0])


S_spec_train_normalized_array = np.array(S_spec_train_normalized)
S_spec_dev_normalized_array = np.array(S_spec_dev_normalized)

num_classes = 2
batch_size = 64
epochs = 10


# convert class vectors to binary class matrices
phq8_array_train = keras.utils.to_categorical(phq8_array_train_extended, num_classes)
phq8_array_dev = keras.utils.to_categorical(phq8_array_dev_extended, num_classes)

#ANN
num_classes = 2
batch_size = 50
epochs = 30

model = Sequential()
model.add(Dense(80, input_shape=(129,ancho), activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(S_spec_train_normalized_array, phq8_array_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(S_spec_dev_normalized_array, phq8_array_dev))

score_train = model.evaluate(S_spec_train_normalized_array, phq8_array_train, verbose=0)
print('Train accuracy:', score_train[1])
score_test = model.evaluate(S_spec_dev_normalized_array, phq8_array_dev, verbose=0)
print('Test accuracy:', score_test[1])
print('Test loss:', score_train[0])
print('Test loss:', score_test[0])


S_spec_train_normalized_array.shape #3D array
phq8_array_train.shape #2D array

#EVALUATION METRICS
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Make predictions on the test dataset
y_pred = model.predict(S_spec_dev_normalized_array)
y_pred_class = np.round(y_pred[:, 0]) # assuming you want to use the first column of the predicted output

# Get the true labels
y_true = phq8_array_dev[:, 0]

# Calculate evaluation metrics
conf_matrix = confusion_matrix(y_true, y_pred_class)
precision = precision_score(y_true, y_pred_class)
recall = recall_score(y_true, y_pred_class)
f1 = f1_score(y_true, y_pred_class)
accuracy = accuracy_score(y_true, y_pred_class)

# Print the evaluation metrics
print("Confusion Matrix: \n", conf_matrix)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)
print("Accuracy: ", accuracy)


#ROC CURVE 1
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Get the true labels
y_true = phq8_array_dev[:, 0]

#Get predicted probabilities
y_pred = model.predict(S_spec_dev_normalized_array)

#Convert predicted probabilities to binary class labels
y_pred_class = np.round(y_pred[:, 0])

#Calculate the AUC of the ROC curve
auc = roc_auc_score(y_true, y_pred_class)

#Calculate the false positive rate and true positive rate for different thresholds
fpr, tpr, thresholds = roc_curve(y_true, y_pred_class)

#Plot the ROC curve
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
