import numpy as np

from models.knn import KNN


def test_knn():
    x = [[0], [0], [4]]
    reg = KNN('regression', n_neighbors=3, weights='distance').fit(x, [2, 6, 30])
    np.testing.assert_allclose(reg.predict([[0]]), [4])
    cls = KNN('classification', n_neighbors=2).fit([[0], [2]], [2, 1])
    np.testing.assert_array_equal(cls.predict([[1]]), [1])


if __name__ == "__main__":
    test_knn()
