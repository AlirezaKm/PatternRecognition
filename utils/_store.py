import pickle
from sys import stderr


def store(filename, data):
    print("[INFO] Storing in file {}".format(filename), file=stderr)
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load(filename):
    print("[INFO] Loading from file {}".format(filename), file=stderr)
    with open(filename, 'rb') as file:
        return pickle.load(file)
