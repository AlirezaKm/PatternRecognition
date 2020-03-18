import numpy as np

from _base import _LearningClass


class KNearestClassifier(_LearningClass):
    EUCLIDEAN_DISTANCE = 'euclidean'

    SUPPORTED_DISTANCE_METHODS = [
        EUCLIDEAN_DISTANCE
    ]

    BRUTE_FORCE_ALGORITHM = 'brute-force'

    SUPPORTED_ALGORITHMS = [
        BRUTE_FORCE_ALGORITHM
    ]

    def __init__(
            self,
            n_neighbor: int,
            X: np.asmatrix,
            y: np.asmatrix,
            dMethod=EUCLIDEAN_DISTANCE,
            algorithm=BRUTE_FORCE_ALGORITHM
    ):
        super(KNearestClassifier, self).__init__(X, y, add_new_feature=False)
        """K-Nearest Neighbor Classifier (KNN)

        :param n_neighbor: number of nearest neighbors
        :param X: Features vector (n x m)
        :param y: Labels vector (m x 1)
        :param dMethod: Distance calculation method (default: 'euclidean')
        :param algorithm: algorithm of choose neighbors (default: 'brute-force')
        """

        self.y = np.asmatrix(self.y, np.int)

        self.n_neighbor: int = n_neighbor
        if self.n_neighbor <= 0:
            raise ValueError('n_neighbor should be greater than 0')

        self.dMethod = dMethod
        if self.dMethod not in self.SUPPORTED_DISTANCE_METHODS:
            raise ValueError('Supported Distance Methods: {}'.format(self.SUPPORTED_DISTANCE_METHODS))

        self.algorithm = algorithm
        if self.algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError('Supported Algorithms: {}'.format(self.SUPPORTED_ALGORITHMS))

    def _distance(self, point_1: np.asmatrix, point_2: np.asmatrix) -> float:

        result = 0
        if self.dMethod == self.EUCLIDEAN_DISTANCE:
            result = np.sqrt(np.sum(np.power(point_1 - point_2, 2), axis=0, dtype=np.float))

        return result

    def predict(self, x_data: np.asmatrix):
        self.check_is_trained()

        result = []
        for n_sample in range(x_data.shape[1]):
            _x = np.full(self.X.shape, np.take(x_data, [n_sample], axis=1))

            # calculate distance from each trainX
            distances = self._distance(self.X, _x)

            # ascending due to their distances
            idx_neighbors = np.array(np.argsort(distances).ravel())[0, :self.n_neighbor]

            # selected nearest neighbors in order to voting
            _selected_y = np.array(self.y[idx_neighbors]).ravel()

            items, counts = np.unique(_selected_y, return_counts=True)

            # zip items with their counts
            items = zip(items, counts)

            # voting
            best, _ = max(items, key=lambda item: item[1])

            # add result to a list
            result.append(best)

        return np.asmatrix(result).T

    def train(self):
        # extract classes
        self.classes = np.unique(self.y, axis=0).ravel()
        self.n_classes = len(self.classes)

        self.set_trained()
        return self
