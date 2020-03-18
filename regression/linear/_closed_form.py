import numpy as np

from _base import _LearningClass


class ClosedFormSolution(_LearningClass):
    def __init__(self, X: np.array, y: np.array):
        """Converge J by Close Form Method

        Pass parameters like this:
            X: n_features x n_samples
            y: n_samples x 1
        """
        super(ClosedFormSolution, self).__init__(X, y)
        self.theta = np.zeros((self.n_features, 1))

    def train(self):
        inv = np.linalg.pinv(np.dot(self.X.T, self.X))
        self.theta = np.dot(np.dot(inv, self.X.T).T, self.y)

        self.set_trained()
        return self

    def predict(self, x_data):
        self.check_is_trained()

        _x_data: np.asmatrix = np.nan_to_num(
            np.vstack([np.ones((1, x_data.shape[1])), x_data])
        )

        return np.nan_to_num(self.theta.T.dot(_x_data).T)
