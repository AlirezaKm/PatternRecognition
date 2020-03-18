import numpy as np

from _base import _LearningClass


class KMeans(_LearningClass):
    EUCLIDEAN_DISTANCE = 'euclidean'

    SUPPORTED_DISTANCE_METHODS = [
        EUCLIDEAN_DISTANCE
    ]

    def __init__(
            self,
            X: np.asmatrix,
            k: int = 5,
            max_iteration: int = 200,
            dMethod=EUCLIDEAN_DISTANCE,
            random_state=None
    ):
        """K-Means Clustering

        :param X: features vector, shape (n_features * n_samples)
        :param k: number of clusters (default: 5)
        :param max_iteration: Maximum Iterations (default: 200)
            TODO: if converged iteration will terminate
        :param: dMethod: Distance calculation method (default: 'euclidean')
        :param random_state: random state for seed
        """
        super(KMeans, self).__init__(X, None, add_new_feature=False)

        self.k: int = k
        if self.k <= 0:
            raise ValueError('k should be greater than 0')

        if self.k > self.n_samples:
            raise ValueError('number of samples lower than clusters')

        self.max_iteration: int = max_iteration
        if self.max_iteration <= 0:
            raise ValueError('max_iteration should be greater than 0')

        self.random_state = random_state
        if self.random_state:
            np.random.seed(random_state)

        self.dMethod = dMethod
        if self.dMethod not in self.SUPPORTED_DISTANCE_METHODS:
            raise ValueError('Supported Distance Methods: {}'.format(
                self.SUPPORTED_DISTANCE_METHODS
            ))

    def _init_centroid(self):
        _idx = np.random.randint(0, self.n_samples, self.k)
        self.init_centers = np.take(self.X, _idx, axis=1)
        self.best_centers = self.init_centers.copy()
        return self.init_centers

    def _distance(self, point_1: np.asmatrix, point_2: np.asmatrix) -> float:

        result = 0
        if self.dMethod == self.EUCLIDEAN_DISTANCE:
            result = np.sqrt(np.sum(np.power(point_1 - point_2, 2), axis=0))

        return result

    def train(self):
        # choose random centroids
        self._init_centroid()

        for iterate in range(self.max_iteration):
            self.center_instances = {_id: [] for _id in range(self.k)}
            for sample in range(self.X.shape[1]):
                _x = np.full(self.best_centers.shape, self.X[:, sample])

                # calculate distance from each trainX
                distances = self._distance(self.best_centers, _x)

                try:
                    # ascending due to their distances
                    best_center_id = np.argsort(distances, axis=1).ravel()[0, 0]

                    # assign sample to center points
                    self.center_instances[best_center_id].append(self.X[:, sample])
                except KeyError as e:
                    print(__file__, e), exit(1)

            # move centroids
            for centroid_id in range(self.k):
                _instance = np.array(self.center_instances[centroid_id])[:, :, 0]
                _center = np.asmatrix(
                    np.mean(_instance, axis=0)
                ).T
                self.best_centers[:, centroid_id] = _center
                self.center_instances[centroid_id] = _instance

        self.set_trained()
        return self

    def predict(self, x_data: np.asmatrix):
        self.check_is_trained()

        x_data = np.asmatrix(x_data)
        if x_data.shape[0] != self.X.shape[0]:
            raise ValueError("x_data doesn't have same dimension with X "
                             "{} != {}".format(x_data.shape, self.X.shape))
        result = []
        for sample in range(x_data.shape[1]):
            _x = np.full(self.best_centers.shape, x_data[:, sample])

            # calculate distance from each trainX
            distances = self._distance(self.best_centers, _x)

            try:
                # ascending due to their distances
                best_center_id = np.argsort(distances, axis=1).ravel()[0, 0]

                # store predicted id
                result.append(best_center_id)
            except KeyError as e:
                print(__file__, e), exit(1)

        return np.asmatrix(result).T
