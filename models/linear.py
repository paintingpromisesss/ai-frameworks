import numpy as np
from sklearn.base import BaseEstimator


def design(x):
    return np.column_stack((np.ones(len(x)), np.asarray(x, dtype=float)))


class LeastSquares(BaseEstimator):
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, x, y):
        a, y = design(x), np.asarray(y, dtype=float)
        if self.alpha > 0:
            penalty = np.sqrt(self.alpha) * np.eye(a.shape[1])[1:]
            a = np.vstack((a, penalty))
            y = np.r_[y, np.zeros(len(penalty))]
        self.coef_ = np.linalg.lstsq(a, y, rcond=None)[0]
        return self

    def predict(self, x):
        return design(x) @ self.coef_


class Logistic(BaseEstimator):
    def __init__(self, C=1.0, max_iter=100, tol=1e-8):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x, y):
        a = design(x)
        self.classes_, target = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2 or self.C <= 0:
            raise ValueError('Binary logistic regression requires two classes and positive C')
        coef = np.zeros(a.shape[1])
        penalty = np.r_[0., np.full(a.shape[1] - 1, 1 / self.C)]

        def loss(w):
            z = a @ w
            return np.sum(np.logaddexp(0, z) - target * z) + 0.5 * np.dot(penalty, w**2)

        for _ in range(self.max_iter):
            p = np.exp(-np.logaddexp(0, -(a @ coef)))
            grad = a.T @ (p - target) + penalty * coef
            if np.linalg.norm(grad, ord=np.inf) <= self.tol:
                break
            hessian = a.T @ ((p * (1 - p))[:, None] * a) + np.diag(penalty)
            step = np.linalg.solve(hessian, grad)
            current = loss(coef)
            scale = 1.
            while scale >= 1e-8 and loss(coef - scale * step) > current:
                scale /= 2
            if scale < 1e-8:
                break
            coef -= scale * step
        self.coef_ = coef
        return self

    def predict(self, x):
        return self.classes_[(design(x) @ self.coef_ > 0).astype(int)]
