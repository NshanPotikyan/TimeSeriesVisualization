import numpy as np
from dtw import DTW


class OneNN:
    """
    One Nearest Neighbor Classifier
    for Time Series Classification (TSC)
    using Dynamic Time Warping (DTW) distance
    """
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.distance_matrix = None

    def fit(self, x_train, y_train):
        """
        Fit the model using x_train as training data and y_train as target values
        :param x_train:
        :param y_train:
        :return:
        """
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        """
        Predict the class labels for the provided data.
        :param x_test: unlabeled time series data
        :return:
        """
        x_train = self.x_train
        n = x_train.shape[0]
        m = x_test.shape[0]
        distance_matrix = np.zeros(shape=(n, m))

        for i in range(n):
            for j in range(m):
                distance_matrix[i, j] = DTW(s1=x_train.iloc[i, :],
                                            s2=x_test.iloc[j, :]).get_distance()

        self.distance_matrix = distance_matrix
        # prediction
        return self.y_train.values[np.argmin(distance_matrix, axis=0)]
