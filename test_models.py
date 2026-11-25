import numpy as np

from models.boosting import Boosting
from models.forest import Forest
from models.knn import KNN
from models.linear import LeastSquares, Logistic
from models.tree import Tree


def test_knn():
    x = [[0], [0], [4]]
    reg = KNN('regression', n_neighbors=3, weights='distance').fit(x, [2, 6, 30])
    np.testing.assert_allclose(reg.predict([[0]]), [4])
    cls = KNN('classification', n_neighbors=2).fit([[0], [2]], [2, 1])
    np.testing.assert_array_equal(cls.predict([[1]]), [1])


def test_linear():
    reg = LeastSquares().fit([[0], [1], [2]], [3, 5, 7])
    np.testing.assert_allclose(reg.predict([[4]]), [11])
    x = [[-2], [-1], [1], [2]]
    cls = Logistic().fit(x, [1, 1, 2, 2])
    np.testing.assert_array_equal(cls.predict(x), [1, 1, 2, 2])


def test_tree():
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [1, 2, 2, 1]
    cls = Tree('classification', max_depth=2).fit(x, y)
    np.testing.assert_array_equal(cls.predict(x), y)
    reg = Tree('regression', min_samples_leaf=2).fit([[0], [1], [2], [3]], [1, 3, 9, 11])
    np.testing.assert_allclose(reg.predict([[0.8], [2.2]]), [2, 10])


def test_forest():
    x = np.arange(20)[:, None]
    y = np.where(x.ravel() < 10, 1, 2)
    cls = Forest('classification', n_estimators=16, random_state=20).fit(x, y)
    np.testing.assert_array_equal(cls.predict([[0], [19]]), [1, 2])
    reg = Forest('regression', n_estimators=8, random_state=20).fit(x, np.full(20, 7))
    np.testing.assert_allclose(reg.predict([[3], [9]]), [7, 7])


def test_boosting():
    x = np.arange(12)[:, None]
    y = np.where(x.ravel() < 6, 3, 13)
    reg = Boosting('regression', n_estimators=20, max_depth=1, random_state=20).fit(x, y)
    np.testing.assert_allclose(reg.predict([[1], [10]]), [3 + 5 * .9**20, 13 - 5 * .9**20])
    cls = Boosting('classification', n_estimators=10, max_depth=1, random_state=20).fit(x, y)
    np.testing.assert_array_equal(cls.predict(x), y)


if __name__ == '__main__':
    test_knn()
    test_linear()
    test_tree()
    test_forest()
    test_boosting()
