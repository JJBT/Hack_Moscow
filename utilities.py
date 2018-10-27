from Constants import *

import numpy as np

import pickle
import scipy.io.wavfile as wav
import wave
import os
import speechpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')

class_labels = ["Neutral", "Angry", "Happy", "Sad"]


def read_wav(filename):
    return wav.read(filename)


def get_train_data(path_to_data, flatten=True, mfcc_len=MFCC_LEN):
    scl = StandardScaler()

    data = []
    labels = []
    cur_dir = os.getcwd()

    os.chdir(ROOT)
    os.chdir(path_to_data)

    for i, directory in enumerate(class_labels):
        os.chdir(directory)
        for filename in os.listdir('.'):
            fs, signal = read_wav(filename)
            s_len = len(signal)
            scl.fit(signal.reshape(-1, 1))
            signal = scl.transform(signal.reshape(-1, 1))
            signal = signal.flatten()

            # pad the signals by zeros to have same size if lesser than required
            # else slice them
            if s_len < MSLEN:
                pad_len = MSLEN - s_len
                pad_rem = pad_len % 2
                pad_len = pad_len // 2
                signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
            else:
                pad_len = s_len - MSLEN
                pad_len = pad_len // 2
                signal = signal[pad_len:pad_len + MSLEN]
            mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)

            if flatten:
                mfcc = mfcc.flatten()

            data.append(mfcc)
            labels.append(i)

        os.chdir('..')

    os.chdir(ROOT)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def display_metrics(y_pred, y_true):
    print(accuracy_score(y_pred=y_pred, y_true=y_true))
    print(confusion_matrix(y_pred=y_pred, y_true=y_true))


def save_model(model):
    with open("models/model.pickle", "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def convert_from_file(filename, flatten, mfcc_len):
    scl = StandardScaler()
    fs, signal = read_wav(filename)
    s_len = len(signal)

    scl.fit(signal.reshape(-1, 1))
    signal = scl.transform(signal.reshape(-1, 1))
    signal = signal.flatten()

    # pad the signals by zeros to have same size if lesser than required
    # else slice them
    if s_len < MSLEN:
        pad_len = MSLEN - s_len
        pad_rem = pad_len % 2
        pad_len = pad_len // 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = s_len - MSLEN
        pad_len = pad_len // 2
        signal = signal[pad_len:pad_len + MSLEN]
    mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)

    if flatten:
        mfcc = mfcc.flatten()

    return np.array(mfcc)


def convert_from_dir(directory, flatten, mfcc_len):
    data = []
    cur_dir = os.getcwd()
    os.chdir(ROOT)
    os.chdir(directory)

    for filename in os.listdir('.'):
        data.append(convert_from_file(filename, flatten, mfcc_len))
    # os.chdir(cur_dir)

    return np.array(data)


def convert_new_data(path, flatten=True, mfcc_len=MFCC_LEN):
    data = convert_from_dir(path, flatten, mfcc_len)
    return data


def decision(data):
    data.apply(lambda x: 1 if x == 1 or x == 3 else 0, inplace=True)

