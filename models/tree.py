from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator


@dataclass
class Node:
    value: object
    feature: int = -1
    threshold: float = 0.
    left: int = -1
    right: int = -1


class Tree(BaseEstimator):
    def __init__(self, task, max_depth=None, min_samples_leaf=1, max_features=None,
                 random_state=None):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        self.nodes_ = []
        if self.task == 'classification':
            self.classes_, encoded = np.unique(y, return_inverse=True)
            target = np.eye(len(self.classes_))[encoded]
        else:
            target = y[:, None].astype(float)

        columns = np.arange(x.shape[1])
        budget = len(columns)
        if self.max_features == 'sqrt':
            budget = max(1, int(np.sqrt(budget)))
        elif isinstance(self.max_features, float):
            budget = max(1, int(budget * self.max_features))
        rng = np.random.default_rng(self.random_state)
        leaf = self.min_samples_leaf

        def grow(rows, depth):
            values = target[rows]
            node = Node(values.mean(axis=0))
            node_id = len(self.nodes_)
            self.nodes_.append(node)
            if (len(rows) < 2 * leaf or (self.max_depth is not None and depth >= self.max_depth)
                    or np.all(values == values[0])):
                return node_id

            # Для one-hot целей уменьшение SSE равно уменьшению взвешенной примеси Джини.
            centered = values - node.value
            total = centered.sum(axis=0)
            best_gain = -np.inf
            features = rng.permutation(columns) if budget < len(columns) or self.random_state is not None else columns
            for checked, feature in enumerate(features, 1):
                col = x[rows, feature]
                order = np.argsort(col, kind='stable')
                v = col[order]
                cuts = np.flatnonzero(v[:-1] < v[1:]) + 1
                cuts = cuts[(cuts >= leaf) & (cuts <= len(rows) - leaf)]
                if len(cuts):
                    left = np.cumsum(centered[order], axis=0)[cuts - 1]
                    right = total - left
                    gains = ((left**2).sum(axis=1) / cuts
                             + (right**2).sum(axis=1) / (len(rows) - cuts))
                    winner = gains.argmax()
                    if gains[winner] > best_gain:
                        best_gain = gains[winner]
                        cut = cuts[winner]
                        node.feature = feature
                        node.threshold = v[cut - 1] + (v[cut] - v[cut - 1]) / 2
                        ordered_rows = rows[order]
                        best_split = ordered_rows[:cut], ordered_rows[cut:]
                if checked >= budget and node.feature >= 0:
                    break
            if node.feature >= 0:
                node.left = grow(best_split[0], depth + 1)
                node.right = grow(best_split[1], depth + 1)
            return node_id

        grow(np.arange(len(x)), 0)
        return self

    def apply(self, x):
        leaves = []
        for row in np.asarray(x, dtype=float):
            index = 0
            node = self.nodes_[0]
            while node.feature >= 0:
                index = node.left if row[node.feature] <= node.threshold else node.right
                node = self.nodes_[index]
            leaves.append(index)
        return np.asarray(leaves, dtype=int)

    def _values(self, x):
        return np.asarray([self.nodes_[i].value for i in self.apply(x)])

    def predict_proba(self, x):
        return self._values(x)

    def predict(self, x):
        values = self._values(x)
        if self.task == 'classification':
            return self.classes_[values.argmax(axis=1)]
        return values.ravel()
