"""Main functions"""

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
from keras.utils import np_utils
import librosa.core
from scipy import stats

import warnings

warnings.filterwarnings('ignore')

class_labels = np.array(["Neutral", "Angry", "Happy", "Sad"])


def read_wav(filename):
    file = librosa.core.load(filename)
    return file[1], file[0]


def get_train_data(path_to_data, flatten=True, mfcc_len=MFCC_LEN):
    """
    Data for training
    """

    # scl = StandardScaler()

    data = []
    labels = []
    weights = []
    cur_dir = os.getcwd()

    os.chdir(ROOT)
    os.chdir(path_to_data)

    for i, directory in enumerate(class_labels):
        os.chdir(directory)
        for filename in os.listdir('.'):
            fs, signal = read_wav(filename)
            s_len = len(signal)

            if filename.find('wg') == -1:
                weights.append(1)
            else:
                weights.append(10)

            # scl.fit(signal.reshape(-1, 1))
            # signal = scl.transform(signal.reshape(-1, 1))
            # signal = signal.flatten()

            # pad the signals by zeros to have same size if lesser than required
            # else slice them
            if s_len < MSLEN:
                pad_len = MSLEN - s_len
                pad_rem = pad_len % 2
                pad_len = pad_len // 2
                signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
            else:
                pad_len = s_len - MSLEN
                pad_rem = pad_len % 2
                pad_len = pad_len // 2
                signal = signal[pad_len:pad_len + MSLEN]

            mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)

            if flatten:
                mfcc = mfcc.flatten()

            data.append(mfcc)
            labels.append(i)

        os.chdir('..')

    os.chdir(ROOT)

    x_train, x_test, y_train, y_test, tr_w, te_w = train_test_split(data, labels, weights, test_size=0.3, random_state=42)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), np.array(tr_w)


def display_metrics(y_pred, y_true):
    print(accuracy_score(y_pred=y_pred, y_true=y_true))
    print(confusion_matrix(y_pred=y_pred, y_true=y_true))


def save_model(model):
    with open("models/model.pickle", "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def convert_from_file(filename, flatten=True, mfcc_len=MFCC_LEN):
    # scl = StandardScaler()
    fs, signal = read_wav(filename)
    s_len = len(signal)

    # scl.fit(signal.reshape(-1, 1))
    # signal = scl.transform(signal.reshape(-1, 1))
    # signal = signal.flatten()

    # pad the signals by zeros to have same size if lesser than required
    # else slice them

    if s_len < MSLEN:
        pad_len = MSLEN - s_len
        pad_rem = pad_len % 2
        pad_len = pad_len // 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = s_len - MSLEN
        pad_rem = pad_len % 2
        pad_len = pad_len // 2
        signal = signal[pad_len:pad_len + MSLEN]

    mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)

    if flatten:
        mfcc = mfcc.flatten()

    return np.array(mfcc)


def convert_from_dir(directory, flatten=True, mfcc_len=MFCC_LEN):
    data = []
    cur_dir = os.getcwd()
    os.chdir(ROOT)
    os.chdir(directory)

    for filename in os.listdir('.'):
        temp = convert_from_file(filename, flatten, mfcc_len)
        if temp is not None:
            data.append(temp)

    # os.chdir(cur_dir)

    return np.array(data)


def convert_new_data(path, flatten=True, mfcc_len=MFCC_LEN):
    data = convert_from_dir(path, flatten, mfcc_len)
    return data


def decision(data):
    data.apply(lambda x: 1 if x == 1 or x == 3 else 0, inplace=True)


def prepare_train_data_dnn(path):
    x_train, x_test, y_train, y_test, weights = get_train_data(path_to_data=path, flatten=False)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return x_train, x_test, y_train, y_test, weights


def prepare_real_data_dnn(path):
    data = convert_new_data(path, flatten=False)

    return np.array(data)


def get_single_pred(model, sample_file, flatten=True):
    # scl = StandardScaler()
    fs, signal = read_wav(sample_file)
    s_len = len(signal)

    segments = []
    predictions = []

    if s_len > MSLEN:
        n = s_len // MSLEN
        rem = s_len % MSLEN

        signal = signal[rem:]
        for i in range(n):
            segments.append(signal[i*MSLEN:(i + 1)*MSLEN])

    # scl.fit(signal.reshape(-1, 1))
    # signal = scl.transform(signal.reshape(-1, 1))
    # signal = signal.flatten()

    for i in segments:
        mfcc = speechpy.feature.mfcc(i, fs, num_cepstral=MFCC_LEN)
        if flatten:
            mfcc = mfcc.flatten()

        mfcc = np.array(mfcc)
        predictions.append(model.predict(mfcc.reshape(1, -1)))

    predictions = np.array(predictions)

    return stats.mode(predictions)[0].flatten()[0]
