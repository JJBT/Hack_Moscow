import pickle
from utilities import convert_from_dir, convert_new_data, get_train_data
from Constants import *


def test():
    with open("models/model.pickle", "rb") as model:
        clf = pickle.load(model)
        print(clf.predict(convert_new_data(PATH_TO_REALDATA)))


if __name__ == '__main__':
    test()
    for i in get_train_data(PATH_TO_DATA)[0]:
        print(i)
        print(len(i))
    print("----")

    print(convert_new_data(PATH_TO_REALDATA))
    print(len(convert_new_data(PATH_TO_REALDATA)[0]))


