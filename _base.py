import abc
import warnings

import numpy as np

# disable warnings
warnings.filterwarnings('ignore')


class _BaseClass:

    def __init__(self, X: np.array, y: np.array = None):
        self.real_X = np.array(X)
        self.X: np.array = np.array(X)

        # n_features, n_samples
        self.n_features, self.n_samples = self.X.shape

        if y is not None:
            self.real_y = np.array(y)
            self.y = np.array(np.nan_to_num(y).reshape(-1, 1))

            self.y_rows, self.y_cols = self.y.shape
            if self.n_samples != self.y_rows:
                raise Exception('y rows != X cols ({} != {})'.format(self.y_rows, self.n_samples))


class _LearningClass(_BaseClass):

    def __init__(self, X: np.array, y: np.array, add_new_feature=True):
        # add Ones to first row
        self.X: np.array = np.nan_to_num(
            np.vstack([np.ones((1, X.shape[1])), X])
        ) if add_new_feature else X

        super(_LearningClass, self).__init__(X, y)

        self._is_trained: bool = False

    def check_is_trained(self):
        assert self._is_trained, 'you should train first'

    def set_trained(self):
        self._is_trained = True

    @abc.abstractmethod
    def predict(self, x_data):
        return

    @abc.abstractmethod
    def train(self):
        pass


class _TransformClass(_BaseClass):

    @abc.abstractmethod
    def transform(self):
        pass
