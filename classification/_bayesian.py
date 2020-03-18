import numpy as np

from _base import _LearningClass


class MultinomialNB(_LearningClass):
    is_trained: bool

    def __init__(self, X: np.array, y: np.array, Laplace: float = 1.0):
        """
        Multinomial Naive Bayes

        :param X: n_features x n_samples
        :param y: n_samples x 1
        :param Laplace: The smoothing prior
        """
        super(MultinomialNB, self).__init__(X, y, add_new_feature=False)

        self.laplace = Laplace
        if self.laplace < 0:
            raise ValueError("Laplace should be > 0 ")

        self.classes = {cls: _id for _id, cls in enumerate(np.unique(self.y, axis=0).ravel())}
        self.n_classes = len(self.classes.keys())
        self._theta = np.asmatrix(np.zeros((self.n_classes, self.n_features)))

        self.priors = np.asmatrix(np.zeros((1, self.n_classes)))
        for cls, _id in self.classes.items():
            self.priors[0, _id] = len(np.where(self.y == cls)[0]) / self.n_samples

    @property
    def class_distribution(self):
        distribution = dict()
        try:
            for cls in self.classes:
                distribution[cls] = len(np.where(self.y == cls)[0]) / self.n_samples

            return distribution
        except KeyError as e:
            print(e)

        return None

    def train(self):
        feature_prior = dict()
        try:
            for cls, _id in self.classes.items():
                indices = np.where(self.y == cls)[0]
                feature_prior[cls] = self.X[:, indices].sum(axis=1) + self.laplace

                self._theta[cls, :] = (np.log(feature_prior[cls]) - np.log(feature_prior[cls].sum())).T

        except Exception as e:
            print('[train]', e)
            exit(-1)

        self.set_trained()
        return self

    def predict(self, x_data: np.asmatrix):
        self.check_is_trained()

        probabilities = np.dot(x_data.T, self._theta.T) + np.log(self.priors)
        return np.argmax(probabilities, axis=1)
