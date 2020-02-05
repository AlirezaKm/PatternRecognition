from _base import _LearningClass
import numpy as np
import sys


class BatchGradientDescent(_LearningClass):

    def __init__(self,
                 X: np.array,
                 y: np.array,
                 theta: np.array = None,
                 alpha: float = 0.0001,
                 iterate: int = 1000,
                 ):
        """Converge J by Gradient Descent Method

        Pass parameters like this:
            theta: (n_features + 1) x 1
            X: n_features x n_samples
            y: n_samples x 1

        @:param X {
            Rows: n_features
            Columns: n_samples
        }
        """
        super(BatchGradientDescent, self).__init__(X, y)

        if theta is None:
            theta = np.asmatrix(np.zeros((self.n_features, 1)))

        self.theta: np.asmatrix = np.asmatrix(theta)
        self.alpha: float = alpha
        self.iterate: int = iterate
        self._cost_history = np.zeros((self.iterate, 1))
        try:
            self.theta_rows = self.theta.shape[0]
        except KeyError as e:
            raise Exception(e)

        if self.n_features != self.theta_rows:
            raise Exception('theta rows != X cols ({} != {})'.format(self.theta_rows, self.n_features))

    @property
    def cost_history(self):
        return self._cost_history

    @property
    def hypothesis(self):
        """
        1 x m
        """

        return np.nan_to_num(np.dot(self.X.T, self.theta))

    @property
    def _cost(self):
        """
        theta: (n_features + 1) x 1
        X: (n_features + 1) x n_samples
        y: n_samples x 1
        """

        h = self.hypothesis

        J = np.dot((h - self.y).T, (h - self.y))
        J = J / (2 * self.n_samples)
        J = np.nan_to_num(J)

        return J

    @property
    def cost(self):
        return self._cost_history[-1]

    def train(self):
        try:

            # Batch Gradient Descent
            _current_itr = None
            for itr in range(self.iterate):

                h = self.hypothesis
                grad = np.nan_to_num(np.dot(self.X, (h - self.y)))

                self.theta = self.theta - (self.alpha * grad / self.n_samples)

                current_cost = self._cost
                self._cost_history[itr] = current_cost
                if (itr >= 1) and \
                        (current_cost == self._cost_history[itr - 1]):
                    if _current_itr is None:
                        _current_itr = itr
                    print("[guid] decrease learning rate (#iterate{})".format(_current_itr), file=sys.stderr)
        except Exception as e:
            print(e, file=sys.stderr)

        self.set_trained()
        return self

    def predict(self, x_data: np.asmatrix):
        self.check_is_trained()

        _x_data: np.asmatrix = np.nan_to_num(
            np.vstack([np.ones((1, x_data.shape[1])), x_data])
        )
        _predict = np.dot(_x_data.T, self.theta)
        return np.nan_to_num(_predict)
