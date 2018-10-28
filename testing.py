"""Testing"""

import pickle
from utilities import convert_from_dir, convert_new_data, get_train_data,\
    prepare_real_data_dnn, read_wav, convert_from_file, get_single_pred, class_labels
from rude_detect import check_text
from Constants import *
from keras.models import load_model
import numpy as np
import os


def test_light_model():
    with open("models/model.pickle", "rb") as model:
        clf = pickle.load(model)

        os.chdir(ROOT)
        os.chdir(PATH_TO_REALDATA)

        for file in os.listdir('.'):
            print(file)
            pred = get_single_pred(clf, file)
            print(pred)


def test_dnn():
    # model = load_model(BEST_MODEL_DNN)

    model = load_model("models/best_model_dnn.h5")

    # model = load_model("models/lstm_eng1_85.h5")
    os.chdir(ROOT)
    print(class_labels[np.argmax(model.predict(prepare_real_data_dnn(PATH_TO_REALDATA)), axis=1)])
    check_text('C:/Users/Vladimyr/PycharmProjects/hack_moscow/dataset/real_data/Angry_Final_Test_1.wav', '../../swear_words.txt')
    check_text('C:/Users/Vladimyr/PycharmProjects/hack_moscow/dataset/real_data/Happy Very Final Test 1.wav', '../../swear_words.txt')
    check_text('C:/Users/Vladimyr/PycharmProjects/hack_moscow/dataset/real_data/Mentor Test.wav', '../../swear_words.txt')


if __name__ == '__main__':
    # test_light_model()
    test_dnn()


