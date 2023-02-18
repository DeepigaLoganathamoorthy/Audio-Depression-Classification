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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from librosa import display
from pydub import AudioSegment
import matplotlib.pyplot as plt
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


S_gfcc_train = ['']*(file_count_train)
S_gfcc_train_normalized = ['']*(file_count_train)
i = 0
for file in os.listdir(train_split):
    if file.endswith(".wav"):
        filepath = os.path.join(train_split, file)
        fs, sig = scipy.io.wavfile.read(filepath)
        gfccs = gfcc(sig, fs=fs, num_ceps=13)
        S_gfcc_train[i] = gfccs
        S_gfcc_train_normalized[i] = tf.math.log(tf.abs(S_gfcc_train[i]) + 0.01)
        i += 1




S_gfcc_dev = ['']*(file_count_dev)
S_gfcc_dev_normalized = ['']*(file_count_dev)
i = 0
for file in os.listdir(dev_split):
    if file.endswith(".wav"):
        filepath = os.path.join(dev_split, file)
        fs, sig = scipy.io.wavfile.read(filepath)
        gfccs = gfcc(sig, fs=fs, num_ceps=13)
        S_gfcc_dev[i] = gfccs	
        S_gfcc_dev_normalized[i] = tf.math.log(tf.abs(S_gfcc_dev[i]) + 0.01)
        i += 1



        
S_gfcc_dev = np.array(S_gfcc_dev)
S_gfcc_dev_normalized = np.log(tf.abs(S_gfcc_dev) + 0.01)
S_gfcc_dev_normalized.shape

S_gfcc_train = np.array(S_gfcc_train)
S_gfcc_train_normalized = np.log(tf.abs(S_gfcc_train) + 0.01)
S_gfcc_train_normalized.shape

S_gfcc_dev_normalized = S_gfcc_dev_normalized.reshape(file_count_dev, -1)
S_gfcc_train_normalized = S_gfcc_train_normalized.reshape(file_count_train, -1)

S_gfcc_dev_normalized.shape
# result 2d array


phq8_array_train_extended = []
phq8_array_dev_extended = []

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



#check
phq8_array_dev_extended.shape
#2d array

#RF

clf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=0)
clf.fit(S_gfcc_train_normalized, phq8_array_train_extended.ravel())
train_accuracy = clf.score(S_gfcc_train_normalized, phq8_array_train_extended.ravel())
print("Train accuracy: ", train_accuracy)
test_accuracy = clf.score(S_gfcc_dev_normalized, phq8_array_dev_extended.ravel())
print("Test accuracy: ", test_accuracy)
predictions = clf.predict_proba(S_gfcc_dev_normalized)
log_loss_value = log_loss(phq8_array_dev_extended.ravel(), predictions)
print("Log loss: ", log_loss_value)


#EVALUATION METRICS

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Make predictions on the test dataset
y_pred = clf.predict(S_gfcc_dev_normalized)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(phq8_array_dev_extended.ravel(), y_pred)
print("Confusion Matrix: \n", conf_matrix)

# Calculate the precision, recall, and F1-score
precision = precision_score(phq8_array_dev_extended.ravel(), y_pred)
recall = recall_score(phq8_array_dev_extended.ravel(), y_pred)
f1 = f1_score(phq8_array_dev_extended.ravel(), y_pred)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)

# Calculate the accuracy
accuracy = accuracy_score(phq8_array_dev_extended.ravel(), y_pred)
print("Accuracy: ", accuracy)

# calculate sensitivity
sensitivity = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
print("Sensitivity: ", sensitivity)

# calculate specificity
specificity = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
print("Specificity: ", specificity)


#ROC CURVE GRAPH
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Make predictions using the decision_function method
y_pred_proba = clf.predict_proba(S_gfcc_dev_normalized)[:,1]

# Compute the ROC-AUC score
roc_auc = roc_auc_score(phq8_array_dev_extended.ravel(), y_pred_proba)

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(phq8_array_dev_extended.ravel(), y_pred_proba)

# Plot the ROC curve
plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(roc_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
