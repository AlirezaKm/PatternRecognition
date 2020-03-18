import numpy as np


def mse(a: np.asmatrix, b: np.asmatrix):
    size_a, size_b = a.shape, b.shape
    if size_a != size_b:
        raise Exception('[MSE] size(a) != size(b) ({} != {})'.format(
            size_a, size_b
        ))

    return np.mean(np.nan_to_num(np.power(a - b, 2)))


def accuracy(y: np.asmatrix, predicted_y: np.asmatrix) -> float:
    # make them matrix
    y_, pred_y_ = np.asmatrix(y), np.asmatrix(predicted_y)
    y_ = y_ if y_.shape[0] > y_.shape[1] else y_.T
    pred_y_ = pred_y_ if pred_y_.shape[0] > pred_y_.shape[1] else pred_y_.T

    y_size, pred_y_size = len(y_), len(pred_y_)
    if y_size < pred_y_size:
        raise Exception('size of y and predicted_y not same {} != {}'.format(
            y_size, pred_y_size
        ))

    equal_items = 0
    for idx in range(y_size):
        if y_[idx] == pred_y_[idx]:
            equal_items += 1

    return equal_items / y_size


def shuffle(X: np.asmatrix, y: np.asmatrix) -> [np.asmatrix,
                                                np.asmatrix]:
    X = np.asmatrix(X)
    y = np.asmatrix(y)

    if y.shape[0] < y.shape[1]:
        raise Exception('y should be vertical !(rows > cols) !({} > {})'.format(*y.shape))

    # Step 1. Get random indexes for each class
    indexes = list(range(0, X.shape[1]))

    # generate random indexes
    np.random.shuffle(indexes)

    X = X[:, indexes]
    y = y[indexes]

    return X, y
