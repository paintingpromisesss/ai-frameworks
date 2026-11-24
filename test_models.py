import numpy as np

from models.linear import LeastSquares, Logistic


def test_linear():
    reg = LeastSquares().fit([[0], [1], [2]], [3, 5, 7])
    np.testing.assert_allclose(reg.predict([[4]]), [11])
    x = [[-2], [-1], [1], [2]]
    cls = Logistic().fit(x, [1, 1, 2, 2])
    np.testing.assert_array_equal(cls.predict(x), [1, 1, 2, 2])


if __name__ == "__main__":
    test_linear()
