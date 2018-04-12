import os
import IPython.display as ipd
import IPython
import pandas as pd
import librosa
import librosa.display
import glob
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import model_selection

PATH = './rec_word_carboidrati/'

# train = np.array([])
train = []
labels = np.array([])

for filename in sorted(os.listdir(PATH)):
    # print(filename)

### set label from file name -> 0 for negative | 1 for positive ###
    sta = filename.rfind("_")
    end = filename.rfind(".")
    label_string = filename[sta+1:end]
    # print('label_string: '+label_string)
    # print(label_string=='positivo')
    if(label_string=='positive'):
        labels = np.append(labels,1)
    else:
        labels = np.append(labels,0)

### read audio data, show in plot, trasform to array ###
    data, sr = librosa.load(PATH+filename)
    # plt.figure(figsize=(12, 4))
    # librosa.display.waveplot(data, sr=sr)
    # plt.title(filename)
    # plt.show()
    # print(data)
    # train = np.append(train,data)
    # print(filename+": "+str(np.shape(data[:55501])))
    train.append(data[:55501])





print(np.shape(train))
# print(train)
# print(labels)

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(train, labels)#, shuffle=False)
print(labels_test)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
accuracy = clf.score(features_test, labels_test)
print("accuracy: "+str(accuracy))
