from _base import _TransformClass
import numpy as np


class PCA(_TransformClass):
    SVD_METHOD = 'svd'
    CO_VARIANCE_METHOD = 'co-variance'
    SUPPORTED_METHODS = [SVD_METHOD, CO_VARIANCE_METHOD]

    def __init__(self, X: np.array, n_components: int, method: str = SVD_METHOD):
        """
        X: m x n
         n: features
         m: samples

        n_components:
         number of components

        method:
         supported methods: 'svd' (default), 'co-variance'

        """
        super(PCA, self).__init__(np.array(X, dtype=np.float64))

        self.n_components: int = n_components
        assert self.n_components <= min(self.n_samples, self.n_features), \
            'n_components={} must be between 0 and min(n_samples, ' \
            'n_features)={}'.format(
                self.n_components,
                min(self.n_samples, self.n_features)
            )

        self.method = method
        assert self.method in self.SUPPORTED_METHODS, \
            'supported methods "{}"'.format(
                self.SUPPORTED_METHODS
            )

    def transform(self) -> np.asmatrix:

        if self.n_features < self.n_components:
            raise Exception('number of components should be ( <= features )')

        # Mean of Data
        self.mean = np.mean(self.X, axis=0)

        # Replace Xi with Xi - mean
        self.X -= self.mean

        if self.method == self.SVD_METHOD:

            # svd function
            U, s, V = np.linalg.svd(self.X, full_matrices=False)

            # store components
            self.components = V[:self.n_components]

            # select components
            V = V.T[:, :self.n_components]

            # Projection
            P = np.dot(self.X, V)

        else:
            """Co-Variance Method"""

            # Covariance
            coVar = np.cov(self.X.T)

            # EigenValues, EigenVectors
            eValues, eVectors = np.linalg.eig(coVar)

            # sort vectors (max to min)
            self.components = eVectors[:, np.argsort(eValues)[::-1][:self.n_components]]

            # Projection
            P = self.components.T.dot(self.X.T)

        return P

    def inv_transform(self):
        return np.dot(self.transform(), self.components) + self.mean
