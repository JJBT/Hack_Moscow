import numpy as np
import sys
import os
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import np_utils
from tqdm import tqdm

from utilities import get_train_data, class_labels
from Constants import *


def get_model(input_shape):

    model = Sequential()

    model.add(LSTM(128, input_shape=(input_shape[0], input_shape[1])))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='tanh'))

    model.add(Dense(len(class_labels), activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model


def evaluate_model(model, x_train, x_test, y_train, y_test):
    # Train the epochs
    best_acc = 0

    # for i in tqdm(range(50)):
    #     p = np.random.permutation(len(x_train))
    #     x_train = x_train[p]
    #     y_train = y_train[p]
    #     model.fit(x_train, y_train, batch_size=32, epochs=1)
    #     loss, acc = model.evaluate(x_test, y_test)
    #     if acc > best_acc:
    #         print('Updated best accuracy', acc)
    #         best_acc = acc
    #         model.save_weights(BEST_MODELS_WEIGHTS_PATH_DNN)
    #
    # model.load_weights(BEST_MODELS_WEIGHTS_PATH_DNN)

    model.fit(x_train, y_train, batch_size=32, epochs=50)

    print('Accuracy = ', model.evaluate(x_test, y_test)[1])
    model.save(BEST_MODEL_DNN)


def start():

    # Read data
    x_train, x_test, y_train, y_test = get_train_data(path_to_data=PATH_TO_DATA, flatten=False)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    os.chdir(ROOT)

    model = get_model(x_train[0].shape)

    evaluate_model(model, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    start()


