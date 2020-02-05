from scipy.optimize import minimize
from _base import _LearningClass
import itertools as it
import numpy as np


class LogisticRegression(_LearningClass):
    # Supported Methods
    BINARY_CLASS = 'binary'
    ONE_VS_ONE_CLASS = 'ovo'
    ONE_VS_ALL_CLASS = 'ova'

    def __init__(self, X: np.array, y: np.array, theta: np.array = None,
                 alpha: float = 1e-4,
                 iteration: int = 1000,
                 multi_class: str = BINARY_CLASS,
                 sigmoid_threshold: float = 0.5):
        """Logistic Regression
        Pass parameters like this:
            :param X: n_features x n_samples
            :param y: n_samples x 1
            :param theta: (n_features + 1) x (number of labels)
            :param multi_class: using multi class (default: 'binary')
                supported values: ['binary', 'ovo', 'ova']
        """
        super(LogisticRegression, self).__init__(X, y)

        # multi classification
        self.multi_class = multi_class

        # encode data
        self._types, self.y = self._encode()

        if self.multi_class == self.BINARY_CLASS:
            self.n_labels = len(self._types.keys()) - 1
        else:
            self.n_labels = len(self._types.keys())

        if theta is None:
            theta = np.asmatrix(np.zeros((self.n_features, self.n_labels)))

        self.sigmoid_threshold = sigmoid_threshold
        self.theta = theta
        self.iterate = iteration
        self.alpha = alpha
        self.cost = np.zeros((self.n_labels, 1))
        try:
            self.theta_rows = self.theta.shape[0]
            self.theta_cols = self.theta.shape[1]
        except KeyError as e:
            raise Exception(e)

        if self.n_features != self.theta_rows:
            raise Exception('theta rows != X cols ({} != {})'.format(self.theta_rows, self.n_features))

        if self.theta_cols != self.n_labels:
            raise Exception('number of labels != theta cols ({} != {})'.format(
                self.n_labels, self.theta_cols
            ))

        if not (0 < self.sigmoid_threshold <= 1):
            raise Exception('threshold should be between (0, 1]')

    def _encode(self) -> [dict, np.asmatrix]:
        """
        Encode Classes to Numbers

        :return: dictionary of mapped classes, encoded_list
        """
        classes = np.unique(np.array(self.y))
        _dict = {idx: cls for idx, cls in enumerate(classes)}

        if (self.multi_class == self.BINARY_CLASS) and len(_dict.keys()) > 2:
            raise Exception("Binary Method don't support 2 > keys.")

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
            item = int(item[0] if self.multi_class == self.ONE_VS_ONE_CLASS else item[0, 0])
            for idx, cls in self._types.items():
                if str(item).lower().strip() == str(idx).lower().strip():
                    _map.append(cls)
                    break

        return np.asmatrix(_map).T

    def sigmoid(self, z):
        """
        m x 1
        """
        return np.nan_to_num(1 / (1 + np.exp(-z)))

    def hypothesis(self, theta=None, X=None):
        """
        (number of labels) x m
        """
        if theta is None:
            theta = self.theta

        if X is None:
            X = self.X

        return self.sigmoid(np.dot(theta.T, X))

    def _cost(self, theta=None, X=None, y=None):

        if theta is None:
            theta = self.theta

        if X is None:
            X = self.X

        if theta is None:
            y = self.y

        m = len(X.T)

        h = self.hypothesis(theta, X)

        J = -(np.dot(np.log(h), y) + np.dot(np.log((1 - h)), 1 - y)).mean()
        grad = self.alpha * np.dot(X, (h.T - y)) / m

        return J, grad

    def train(self):
        if self.multi_class == self.BINARY_CLASS:
            # minimizer function
            result = minimize(
                fun=self._cost, x0=self.theta, args=(self.X, self.y),
                method='TNC',
                jac=True,
                options={
                    'maxiter': self.iterate,
                    'disp': False
                })
            self.cost = result.fun
            self.theta = np.asmatrix(result.x).T

        elif self.multi_class == LogisticRegression.ONE_VS_ALL_CLASS:
            for label in range(self.n_labels):
                _y = (self.y == label).astype(np.int)
                result = minimize(
                    fun=self._cost, x0=self.theta[:, label], args=(self.X, _y),
                    method='TNC',
                    jac=True,
                    options={
                        'maxiter': self.iterate,
                        'disp': False
                    })

                self.theta[:, label] = np.asmatrix(result.x).T
                self.cost[label] = result.fun

        elif self.multi_class == LogisticRegression.ONE_VS_ONE_CLASS:
            self.subsets = list(it.combinations(set(self._types.keys()), 2))
            for idx, sub in enumerate(self.subsets):
                # we should exclusive
                valid_indexes = np.isin(self.y, list(sub))
                X = self.X[:, valid_indexes.ravel() == 1]
                y = self.y[valid_indexes.ravel() == 1]

                # OneHot
                y = np.asmatrix(y == sub[1]).astype(np.int)

                # minimizer function
                result = minimize(
                    fun=self._cost, x0=self.theta[:, idx], args=(X, y),
                    method='TNC',
                    jac=True,
                    options={
                        'maxiter': self.iterate,
                        'disp': False
                    })

                self.theta[:, idx] = np.asmatrix(result.x).T
                self.cost[idx] = result.fun

        self.set_trained()
        return self

    def predict(self, x_data, encoded: bool = False):
        self.check_is_trained()

        _x_data: np.asmatrix = np.nan_to_num(
            np.vstack([np.ones((1, x_data.shape[1])), x_data])
        )
        result = self.hypothesis(X=_x_data).T

        if self.multi_class == self.BINARY_CLASS:
            "Lazy Method"
            # TODO: refactor it to replace previous names for result same as ONE_VS_ALL_METHOD
            one_positions, _ = np.where(result >= self.sigmoid_threshold)
            result[one_positions] = 1

            zero_positions, _ = np.where(result < self.sigmoid_threshold)
            result[zero_positions] = 0

        elif self.multi_class == self.ONE_VS_ALL_CLASS:
            result = result.argmax(axis=1)
            if not encoded:
                result = self._decode(result)

        elif self.multi_class == self.ONE_VS_ONE_CLASS:

            _result = (result >= self.sigmoid_threshold).astype(np.int)
            _final_result = np.zeros((_x_data.shape[1], 1))
            for sample in range(_x_data.shape[1]):
                _counts = {item: 0 for item in range(self.n_labels)}
                for classifier in range(self.n_labels):
                    _result[sample, classifier] = self.subsets[classifier][_result[sample, classifier]]
                    _counts[_result[sample, classifier]] += 1
                _final_result[sample] = np.argmax(list(_counts.values()), axis=0)

            result = _final_result
            if not encoded:
                result = self._decode(result)

        return result
