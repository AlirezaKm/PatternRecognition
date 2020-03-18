import numpy as np

from _base import _LearningClass


class SoftmaxRegression(_LearningClass):

    def __init__(self, X: np.array, y: np.array, theta: np.array = None,
                 alpha: float = 1e-2,
                 iteration: int = 1000):
        """Softmax Regression
        Pass parameters like this:
            X: n_features x n_samples
            y: n_samples x 1
            theta: (n_features + 1) x (number of labels)
        """
        super(SoftmaxRegression, self).__init__(X, y)

        # encode data
        self._types, self.y = self._encode()

        self.n_labels = len(self._types.keys())

        if theta is None:
            theta = np.asmatrix(np.zeros((self.n_features, self.n_labels)))

        self.theta = np.asmatrix(theta)
        self.iterate = iteration
        self.alpha = alpha
        self.cost_h = []
        try:
            self.theta_rows, self.theta_cols = self.theta.shape
        except KeyError as e:
            raise Exception(e)

        if self.n_features != self.theta_rows:
            raise Exception('theta rows != X cols ({} != {})'.format(self.theta_rows, self.n_features))

        if self.theta_cols != self.n_labels:
            raise Exception('number of labels != theta cols ({} != {})'.format(
                self.n_labels, self.theta_cols
            ))

    def _encode(self) -> [dict, np.asmatrix]:
        """
        Encode Classes to Numbers

        :return: dictionary of mapped classes, encoded_list
        """
        classes = np.unique(np.array(self.y))
        _dict = {idx: cls for idx, cls in enumerate(classes)}
        _map = list()
        for item in self.y:
            for idx, cls in enumerate(classes):
                if str(item[0, 0]).lower().strip() == str(cls).lower().strip():
                    _map.append(idx)
                    break
        result = np.asmatrix(_map).T

        return _dict, result

    def _decode(self, data):

        _map = list()
        for item in data:
            for idx, cls in self._types.items():
                if str(item[0, 0]).lower().strip() == str(idx).lower().strip():
                    _map.append(cls)
                    break

        return np.asmatrix(_map).T

    def softmax(self, z):
        """
        m x (number of labels)
        """
        res = np.exp(z) / np.sum(np.exp(z), axis=1)
        return np.nan_to_num(res)

    def hypothesis(self, theta=None, X=None):
        """
        (number of labels) x m
        """
        if theta is None:
            theta = self.theta

        if X is None:
            X = self.X

        return self.softmax(np.dot(theta.T, X))

    def train(self):
        for itr in range(self.iterate):
            J, theta = 0, np.asmatrix(np.zeros(self.theta.shape))
            for label in range(self.n_labels):
                # OneHot
                y = (self.y == label).astype(np.int)

                # softmax
                h = np.nan_to_num(
                    np.exp(np.dot(self.theta[:, label].T, self.X))
                    / np.sum(np.exp(np.dot(self.theta.T, self.X)), axis=0)
                )

                # cost
                J += -(np.dot(np.log(h), y))

                # gradient
                grad = -np.dot(self.X, y - h.T)

                # temporary theta
                theta[:, label] = self.alpha * grad

            self.theta -= theta / self.n_samples
            J /= self.n_samples
            self.cost_h.append(J[0, 0])

        self.set_trained()
        return self

    def predict(self, x_data, encode: bool = False):
        self.check_is_trained()

        _x_data: np.asmatrix = np.nan_to_num(
            np.vstack([np.ones((1, x_data.shape[1])), x_data])
        )

        result = self.hypothesis(X=_x_data).T
        result = result.argmax(axis=1)
        if not encode:
            result = self._decode(result)

        return result
