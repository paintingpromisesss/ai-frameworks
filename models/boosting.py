import numpy as np
from sklearn.base import BaseEstimator

from models.tree import Tree


class Boosting(BaseEstimator):
    def __init__(self, task, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.task = task
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y)
        classification = self.task == 'classification'
        if classification:
            self.classes_, y = np.unique(y, return_inverse=True)
            if len(self.classes_) != 2:
                raise ValueError('Binary boosting requires two classes')
        mean = y.mean()
        self.initial_ = np.log(mean / (1 - mean)) if classification else mean
        score = np.full(len(y), self.initial_, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.stages_ = []
        for _ in range(self.n_estimators):
            prediction = np.exp(-np.logaddexp(0, -score)) if classification else score
            residual = y - prediction
            tree = Tree('regression', max_depth=self.max_depth, random_state=int(rng.integers(2**32)))
            tree.fit(x, residual)
            leaves = tree.apply(x)
            size = len(tree.nodes_)
            numerator = np.bincount(leaves, weights=residual, minlength=size)
            curvature = prediction * (1 - prediction) if classification else None
            denominator = np.bincount(leaves, weights=curvature, minlength=size)
            updates = np.divide(numerator, denominator, out=np.zeros(size), where=denominator > 1e-12)
            score += self.learning_rate * updates[leaves]
            self.stages_.append((tree, updates))
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        score = np.full(len(x), self.initial_, dtype=float)
        for tree, updates in self.stages_:
            score += self.learning_rate * updates[tree.apply(x)]
        if self.task == 'classification':
            return self.classes_[(score > 0).astype(int)]
        return score
