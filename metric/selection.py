from . import accuracy, mse, shuffle
import numpy as np


def train_test_split(
        X: np.array,
        y: np.array,
        test_size: float = 0.2,
        shuff: bool = True,
        rand_state: float = None):
    X = np.array(X)
    y = np.array(y)

    x_n = X.shape[0]
    x_m = X.shape[1]

    y_m = y.shape[0]
    y_n = y.shape[1]

    if y_n > y_m:
        raise Exception('y matrix should be vertical')

    if x_m != y_m:
        raise Exception('number of samples in X and y is not equal')

    if not (0 < test_size < 1):
        raise Exception('test_size should be between (0, 1)')

    if rand_state is not None:
        np.random.seed(rand_state)

    if shuff:
        X, y = shuffle(X, y)

    # classification
    if y.dtype == np.object:

        train_size = 1 - test_size

        # classes
        classes = np.unique(np.array(y))

        # find indexes of each class
        cls_idx = {cls: np.where(y == cls)[0] for cls in classes}

        # number of test data for each classes
        num_of_split_classes = {
            cls: int(np.ma.ceil(len(cls_idx[cls]) * train_size)) for cls in classes
        }

        # initialize variables
        x_train, x_test = None, None
        y_train, y_test = None, None

        # concatenate elements of each class
        for idx, cls in enumerate(classes):
            # separate train and test data
            _x_test = X.T[cls_idx[cls][num_of_split_classes[cls]:]]
            _y_test = y[cls_idx[cls][num_of_split_classes[cls]:]]

            _x_train = X.T[cls_idx[cls][:num_of_split_classes[cls]]]
            _y_train = y[cls_idx[cls][:num_of_split_classes[cls]]]

            # stupid method !
            if idx == 0:
                # data initializer
                x_train, x_test = _x_train, _x_test
                y_train, y_test = _y_train, _y_test
            else:
                # test data
                x_test = np.concatenate((x_test, _x_test))
                y_test = np.concatenate((y_test, _y_test))

                # train data
                x_train = np.concatenate((x_train, _x_train))
                y_train = np.concatenate((y_train, _y_train))

        # transpose test and train data
        x_train = x_train.T
        x_test = x_test.T

    # linear regression
    else:
        cut = np.ceil(x_m * test_size)

        # test data split
        x_test = (X.T[:cut]).T
        y_test = y[:cut]

        # train data split
        x_train = (X.T[cut:]).T
        y_train = y[cut:]

    return x_train, x_test, y_train, y_test


class KFold:

    def __init__(
            self,
            clf,
            X: np.array,
            y: np.array,
            k: int = 10,
            times: int = 10,
            shuff: bool = False,
            random_state: int = None,
            execute: bool = False,
            mse: bool = True,
            train_score_enable: bool = False,
            **args
    ):
        """
        n-Times K-Fold Cross Validation

        :param clf: Classifier
        :param X: Features Vector
        :param y: Labels Vector
        :param k: number of cross validation (default: 10)
        :param times: times (default: 10)
        :param shuff: shuffle (default: False)
        :param random_state: seed of random generator (default: None)
        :param execute: run K-Fold after initialization (default: False)
        :param train_score_enable: calculate evaluations for train data (default: False)
        :param **args: Classifier arguments
        """

        self.X, self.y = np.array(X), np.array(y).reshape(-1, 1)

        self.n, self.m = self.n_features, self.n_samples = X.shape[0], X.shape[1]

        if self.n_samples != self.y.shape[0]:
            raise Exception('number of samples in X and y is not equal ({} != {})'.format(
                self.n_samples, self.y.shape[0]
            ))

        self.k: int = k
        if self.k <= 1:
            raise Exception('k should be larger than 1')

        self.times: int = times
        if self.times < 1:
            raise Exception('times should be larger than 0')

        if random_state is not None:
            np.random.seed(random_state)

        # classes
        self.classes: np.array = (np.unique(self.y) if self.y.dtype == np.object else np.unique(self.y, axis=0)).ravel()

        # classifier
        self.clf = clf
        self.clf_args = args

        # enable shuffle or not
        self._shuffle: bool = shuff
        if self._shuffle:
            np.random.seed(random_state)

        self.train_score_enable: bool = train_score_enable

        # enable calculate MSE
        self.mse: bool = mse
        self.all_mse: list = []
        self.best_mse = 0

        # estimator
        self.estimators: list = []
        self.best_estimator_ = None

        # scores
        self.best_score: float = 0
        self.test_mean_score: float = 0
        self.test_global_scores: list = []
        self.train_global_scores: list = []

        # best collection
        self.best_x_train = self.best_x_test = self.best_y_train = self.best_y_test = None
        self.best_collection = [self.best_x_train, self.best_x_test, self.best_y_train, self.best_y_test]

        if execute:
            self.run()

    def run(self):

        for _ in range(self.times):

            if self._shuffle:
                self.X, self.y = shuffle(self.X, self.y)

            # find indexes of each class
            cls_idx = {cls: np.where(self.y == cls)[0] for cls in self.classes}
            cv_list = {_id: list() for _id in range(self.k)}

            # create a list for each cv lists
            for cls in self.classes:
                # split array
                part_cv = np.array_split(cls_idx[cls], self.k)

                # mix selected indexes of each class for cross validation lists
                for _id in range(self.k):
                    cv_list[_id].extend(part_cv[_id])

            # TODO: novelty
            # # balance cv lists
            # cv_final = []
            # for _cv_list in cv_list.values():
            #     cv_final.extend(_cv_list)
            #
            # # final list of test list indexes
            # cv_final = np.array_split(cv_final, self.k)

            # accuracy list
            test_local_scores, train_local_scores = [], []
            for _cv_id in range(self.k):
                indexes = cv_list[_cv_id]

                # train data
                x_train, y_train = np.delete(self.X, indexes, axis=1), np.delete(self.y, indexes, axis=0)

                # test data
                x_test, y_test = np.take(self.X, indexes, axis=1), np.take(self.y, indexes, axis=0)

                # train
                classifier = self.clf(
                    X=x_train,
                    y=y_train,
                    **self.clf_args
                ).train()

                # prediction

                # X[:, indexes]
                predicted_y = classifier.predict(x_test)

                # score
                score = accuracy(y_test, predicted_y)

                # y[indexes, :]
                test_local_scores.append(score)

                # calculate train scores
                if self.train_score_enable:
                    # prediction

                    # X[:, indexes]
                    train_predicted_y = classifier.predict(x_train)

                    # score
                    train_score = accuracy(y_train, train_predicted_y)

                    # y[indexes, :]
                    train_local_scores.append(train_score)

                _mse = mse(y_test, predicted_y) if self.mse else 0
                self.all_mse.append(_mse)

                # find best estimator and best score
                if self.best_score < score:

                    self.best_score = score
                    self.best_estimator_ = classifier

                    # collection
                    self.best_x_train, self.best_x_test = x_train, x_test
                    self.best_y_train, self.best_y_test = y_train, y_test

                    # mse
                    if self.mse:
                        self.best_mse = _mse

            self.test_global_scores.append(np.mean(test_local_scores))
            self.train_global_scores.append(np.mean(train_local_scores))

        self.test_mean_score = np.mean(self.test_global_scores)
        self.train_mean_score = np.mean(self.train_global_scores)

        return self

    def __str__(self):
        return "{}".format(self.best_score)