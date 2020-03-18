from typing import List

import cvxopt
import numpy as np

from _base import _LearningClass


class SVM(_LearningClass):
    LINEAR_KERNEL = 'linear'
    RBF_KERNEL = 'rbf'
    POLYNOMIAL_KERNEL = 'polynomial'
    SUPPORTED_KERNELS = [
        LINEAR_KERNEL,
        RBF_KERNEL,
        POLYNOMIAL_KERNEL
    ]

    MULTICLASS_OVA = 'ova'
    MULTICLASS_OVR = 'ovr'
    SUPPORTED_METHODS = [
        MULTICLASS_OVA, MULTICLASS_OVR
    ]

    LABELS = [1, -1]

    def __init__(
            self,
            X: np.asmatrix,
            y: np.asmatrix,
            kernel=LINEAR_KERNEL,
            C: float = None,
            gamma: float = None,
            a: float = 1,
            b: int = 3,
            multiclass=MULTICLASS_OVA,
            non_zero: float = 1e-5,
            encode_data: bool = True,
            verbose: bool = False,
            iteration=1000,
    ):
        """
        SVM
        An implementation of Support Vector Machine (SVM)

        :param X: features (n, m)

        :param y: labels (m, 1)

        :param kernel: Kernel (default: linear)
            Supported Kernels: ['linear', 'rbf', 'polynomial']

        :param C: Regularization Param (default: None)
            lower  C: Soft Margin
            higher C: Hard Margin

        :param gamma: Radial Basis Function (RBF) parameter (default: 1 / (n_features * X.var()))

        :param a: polynomial parameter, bias (default: 1.0)

        :param b: polynomial parameter, power (default: 2)

        :param multiclass: Classification Method (default: 'ova')
            Supported Methods: ['ova' or 'ovr']

        :param non_zero: threshold of non-zero (default: 1e-5)

        :param encode_data: Encode Labels (default: False)

        :param verbose: Verbose Optimisation
        """
        super(SVM, self).__init__(X, y, add_new_feature=False)

        if self.X.dtype not in [np.object, np.str, np.bool]:
            self.X = self.X.astype(np.float64)

        if self.y.dtype not in [np.object, np.str, np.bool]:
            self.y = self.y.astype(np.float64)

        self.iteration: int = iteration
        self.verbose: bool = verbose
        self.non_zero: float = non_zero

        self.multiclass: str = multiclass
        if self.multiclass not in self.SUPPORTED_METHODS:
            raise ValueError('Supported Methods: "{}"'.format(self.SUPPORTED_METHODS))

        self._kernel_method: str = kernel
        if self._kernel_method not in self.SUPPORTED_KERNELS:
            raise ValueError('Supported Kernels: "{}"'.format(self.SUPPORTED_KERNELS))

        # choose kernel function
        self.poly_a: float = a
        self.poly_b: int = b
        self.C: float = C
        if self.C is not None and self.C < 0:
            raise ValueError('C must be positive')
        self.gamma: float = 1 / (self.n_features * self.X.var()) if gamma is None else gamma
        if self._kernel_method == self.LINEAR_KERNEL:
            self.kernel_func = self._linear_kernel
        elif self._kernel_method == self.RBF_KERNEL:
            self.kernel_func = self._gaussian_kernel
        elif self._kernel_method == self.POLYNOMIAL_KERNEL:
            self.kernel_func = self._polynomial_kernel

        self.classes: np.array = (np.unique(self.y) if self.y.dtype == np.object else np.unique(self.y, axis=0)).ravel()
        self.n_classes = len(self.classes)
        if self.n_classes < 2:
            raise ValueError("'y' has 1 class (you should have more than 1 class)")
        self._is_binary = True if self.n_classes == 2 else False

        self.encode_data: bool = encode_data
        if self.encode_data and self._is_binary:
            self.y = self._encode(self.y)

        # future variables
        self.alphas: List[np.array] = []
        self.support_vectors_indexes: List[np.array] = []
        self.support_vectors: List[np.array] = []
        self.support_vectors_y: List[np.array] = []

        # each row contains weights due to features of each classifier
        self.w: np.array = np.zeros((self.n_classes, self.n_features))

        # each row is intercept of a classifier
        self.intercepts: np.array = np.zeros((self.n_classes, 1))

    def _encode(self, y: np.asmatrix) -> np.asmatrix:
        y_ = np.zeros(y.shape).astype(np.float64)
        mapper = {idx: np.where(y == cls)[0] for idx, cls in enumerate(self.classes)}
        for idx, indexes in mapper.items():
            y_[indexes] = self.LABELS[idx]

        return y_

    def _decode(self, y: np.asmatrix) -> np.asmatrix:
        y_ = np.zeros(y.shape).astype(np.float64)
        mapper = {idx: np.where(y == cls)[0] for idx, cls in enumerate(self.LABELS)}
        for idx, indexes in mapper.items():
            y_[indexes] = self.classes[idx]

        return y_

    def _linear_kernel(self, x_0: np.asmatrix, x_1: np.asmatrix):
        return np.linalg.multi_dot([x_0, x_1])

    def _gaussian_kernel(self, x_0: np.asmatrix, x_1: np.asmatrix = None):
        return np.exp((-np.linalg.norm(x_0 - x_1) ** 2) * self.gamma)

    def _polynomial_kernel(self, x_0: np.asmatrix, x_1: np.asmatrix):
        return np.power(self.poly_a + np.linalg.multi_dot([x_0, x_1]), self.poly_b)

    def _train(self, X_, y_):
        """
        we optimise svm with CVXOPT library
        https://cvxopt.org/userguide/coneprog.html#quadratic-programming
        """

        # P
        P = cvxopt.matrix((np.outer(y_, y_) * self.kernel), (self.n_samples, self.n_samples))

        # q
        q = cvxopt.matrix(np.ones((self.n_samples, 1)) * -1)

        # G
        G_arg = np.vstack((np.eye(self.n_samples) * -1, np.eye(self.n_samples))) if self.C \
            else np.eye(self.n_samples) * -1
        G = cvxopt.matrix(G_arg)

        # h
        h_arg = np.hstack((np.zeros(self.n_samples), np.ones(self.n_samples) * self.C)) if self.C \
            else np.zeros(self.n_samples)
        h = cvxopt.matrix(h_arg)

        # A
        A = cvxopt.matrix(y_.reshape(1, -1))

        # b
        _b = cvxopt.matrix(np.zeros(1))

        # options
        cvxopt.solvers.options['show_progress'] = self.verbose
        cvxopt.solvers.options['iterations'] = self.iteration
        cvxopt.solvers.options['debug'] = self.verbose

        # Lagrange
        optimise = cvxopt.solvers.qp(P, q, G, h, A, _b, kktsolver='chol2')

        alphas = np.ravel(optimise.get('x', []))
        if len(alphas) == 0:
            raise ValueError('alphas is empty')

        # indexes
        support_vectors_indexes = np.arange(
            len(alphas)
        )[alphas > self.non_zero]

        # alpha: lagrange multipliers
        alpha = alphas[support_vectors_indexes]

        # support vectors
        support_vectors = X_[:, support_vectors_indexes]
        support_vectors_y = y_[support_vectors_indexes]

        # weights
        if self._kernel_method == self.LINEAR_KERNEL:
            w = np.sum([
                alpha[n] * support_vectors_y[n] * support_vectors[:, n]
                for n in range(len(alpha))
            ], axis=0).ravel()
        else:
            w = np.zeros(1)

        # Intercept
        B = np.sum([
            support_vectors_y[i] - np.sum(
                alpha * support_vectors_y.ravel() * self.kernel[support_vectors_indexes[i], support_vectors_indexes]
            ) for i in range(len(alpha))
        ]) / len(alpha)

        return support_vectors_indexes, support_vectors_y, support_vectors, alpha, w, B

    def train(self):
        # Kernel
        if self._kernel_method == self.LINEAR_KERNEL:
            self.kernel = self.kernel_func(self.X.T, self.X)
        else:
            self.kernel = np.zeros((self.n_samples, self.n_samples))
            for row in range(self.n_samples):
                for col in range(self.n_samples):
                    self.kernel[row, col] = self.kernel_func(self.X[:, row].T, self.X[:, col])

        if self._is_binary:
            self.support_vectors_indexes, self.alphas, \
            self.support_vectors, self.support_vectors_y, \
            self.w, self.intercepts = self._train(self.X, self.y)
        else:
            for idx, cls in enumerate(self.classes):
                # TODO: we need to refactor it

                y_ = np.zeros(self.y.shape)
                _mapper = {
                    '+': np.where(self.y == cls)[0],
                    '-': np.where(self.y != cls)[0]
                }
                y_[_mapper['+']], y_[_mapper['-']] = self.LABELS

                support_vectors_indexes, support_vectors_y, \
                support_vectors, alpha, w, B = self._train(self.X, y_)
                self.support_vectors_indexes.append(support_vectors_indexes)
                self.support_vectors_y.append(support_vectors_y)
                self.support_vectors.append(support_vectors)
                self.w[idx, :] = w.flatten()
                self.intercepts[idx] = B.flatten()
                self.alphas.append(alpha)

        self.set_trained()
        return self

    def decision_function(self, x_data: np.asmatrix) -> np.asmatrix:
        self.check_is_trained()

        n_samples = x_data.shape[1]
        X_ = np.asmatrix(x_data)
        if X_.shape[0] != self.n_features:
            raise ValueError('(x_data features) {} != {} (X features)'.format(
                X_.shape[0], self.n_features
            ))

        def _linear_kernel(w, intercept) -> np.array:
            _w = np.repeat(np.asmatrix(w).T, n_samples, axis=1)
            return np.sum(np.multiply(X_, _w), axis=0).T + intercept

        def _non_linear_kernel(
                alphas, support_vectors_y,
                support_vectors, intercept
        ) -> np.array:
            y_ = np.zeros((n_samples, 1))
            for sample in range(n_samples):
                predicted_ = 0
                for alpha, support_vector_y, support_vector in zip(
                        alphas, support_vectors_y, support_vectors.T
                ):
                    predicted_ += alpha * support_vector_y * self.kernel_func(X_[:, sample].T, support_vector)
                y_[sample] = predicted_

            return y_ + intercept

        def _binary(kmethod):
            return _linear_kernel(self.w, self.intercepts) if kmethod == self.LINEAR_KERNEL \
                else _non_linear_kernel(self.alphas, self.support_vectors_y, self.support_vectors, self.intercepts)

        def _multi_class(kmethod):

            if kmethod == self.LINEAR_KERNEL:
                res = np.zeros((self.n_classes, n_samples))
                for idx in range(self.n_classes):
                    res[idx, :] = _linear_kernel(self.w[idx, :], self.intercepts[idx]).flatten()
            else:
                res = np.zeros((self.n_classes, n_samples))
                for idx in range(self.n_classes):
                    res[idx, :] = _non_linear_kernel(
                        self.alphas[idx], self.support_vectors_y[idx],
                        self.support_vectors[idx], self.intercepts[idx]
                    ).flatten()

            return res

        return _binary(self._kernel_method) if self._is_binary else _multi_class(self._kernel_method)

    def predict(self, x_data: np.asmatrix):
        self.check_is_trained()

        result = self.decision_function(x_data)
        if self._is_binary:
            y = np.sign(result) if result.dtype not in [np.object, np.str] else result
            result = self._decode(y) if self.encode_data else y
        else:
            if self.multiclass in [self.MULTICLASS_OVA, self.MULTICLASS_OVR]:
                y = np.argmax(result, axis=0)
                result = self.classes.take(np.asarray(y, dtype=np.intp))

        return result
