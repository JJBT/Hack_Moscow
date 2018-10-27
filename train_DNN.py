import numpy as np
import sys
import os
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv2D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D
from keras.utils import np_utils
from tqdm import tqdm

from utilities import get_train_data, class_labels
from Constants import *

models = ["CNN", "LSTM"]


def get_model(model_name, input_shape):

    model = Sequential()
    if model_name == 'CNN':
        model.add(Conv2D(8, (13, 13),
                         input_shape=(input_shape[0], input_shape[1], 1)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(8, (13, 13)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Conv2D(8, (13, 13)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(8, (2, 2)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

    elif model_name == 'LSTM':
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

    for i in tqdm(range(50)):
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        y_train = y_train[p]
        model.fit(x_train, y_train, batch_size=32, epochs=1)
        loss, acc = model.evaluate(x_test, y_test)
        if acc > best_acc:
            print('Updated best accuracy', acc)
            best_acc = acc
            model.save_weights(BEST_MODEL_PATH_DNN)

    model.load_weights(BEST_MODEL_PATH_DNN)
    print('Accuracy = ', model.evaluate(x_test, y_test)[1])


def start(n):
    """
    :param n: 1-CNN, 2-LSTM
    """
    print('model given', models[n])

    # Read data
    x_train, x_test, y_train, y_test = get_train_data(path_to_data=PATH_TO_DATA, flatten=False)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    os.chdir(ROOT)

    if n == 0:
        # Model is CNN so have to reshape the data
        in_shape = x_train[0].shape
        x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)

    model = get_model(models[n], x_train[0].shape)

    evaluate_model(model, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    n = int(sys.argv[1]) - 1
    start(n)


