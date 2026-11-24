import numpy as np
from sklearn.base import BaseEstimator


class KNN(BaseEstimator):
    def __init__(self, task, n_neighbors=5, weights='uniform', p=2):
        self.task = task
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, x, y):
        self.x_ = np.asarray(x, dtype=float)
        self.y_ = np.asarray(y)
        if not 1 <= self.n_neighbors <= len(self.x_):
            raise ValueError('n_neighbors must be between 1 and the training size')
        if self.task == 'classification':
            self.classes_, self.y_ = np.unique(y, return_inverse=True)
        return self

    def predict(self, x):
        predictions = []
        for row in np.asarray(x, dtype=float):
            distance = (np.abs(self.x_ - row) ** self.p).sum(axis=1) ** (1 / self.p)
            neighbors = np.argsort(distance, kind='stable')[:self.n_neighbors]
            d = distance[neighbors]
            values = self.y_[neighbors]
            if self.weights == 'uniform':
                weight = np.ones_like(d)
            elif np.any(d == 0):
                weight = (d == 0).astype(float)
            else:
                weight = 1 / d
            if self.task == 'classification':
                votes = np.bincount(values, weights=weight, minlength=len(self.classes_))
                predictions.append(self.classes_[votes.argmax()])
            else:
                predictions.append(np.average(values, weights=weight))
        return np.asarray(predictions)
