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

PATH = './rec_word_carboidrati/'

train = np.array([])
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
    train = np.append(train,data)



print(train)
print(labels)


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(train, labels)
# clf.fit(feature_training, label_training)
# prediction = clf.predict(feature_test)
# accuracy = clf.score(feature_test, label_test)
