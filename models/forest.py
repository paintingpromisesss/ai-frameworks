import numpy as np
from sklearn.base import BaseEstimator

from models.tree import Tree


class Forest(BaseEstimator):
    def __init__(self, task, n_estimators=100, max_features=None, min_samples_leaf=1,
                 random_state=None):
        self.task = task
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        max_features = self.max_features
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            if max_features is None:
                max_features = 'sqrt'
        for _ in range(self.n_estimators):
            rows = rng.integers(0, len(x), size=len(x))
            tree = Tree(self.task, max_features=max_features, min_samples_leaf=self.min_samples_leaf,
                        random_state=int(rng.integers(2**32)))
            self.trees_.append(tree.fit(x[rows], y[rows]))
        return self

    def predict(self, x):
        if self.task == 'regression':
            return np.mean([tree.predict(x) for tree in self.trees_], axis=0)
        totals = np.zeros((len(x), len(self.classes_)))
        for tree in self.trees_:
            columns = np.searchsorted(self.classes_, tree.classes_)
            totals[:, columns] += tree.predict_proba(x)
        return self.classes_[totals.argmax(axis=1)]
