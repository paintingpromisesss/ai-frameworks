import numpy as np

from models.forest import Forest


def test_forest():
    x = np.arange(20)[:, None]
    y = np.where(x.ravel() < 10, 1, 2)
    cls = Forest('classification', n_estimators=16, random_state=20).fit(x, y)
    np.testing.assert_array_equal(cls.predict([[0], [19]]), [1, 2])
    reg = Forest('regression', n_estimators=8, random_state=20).fit(x, np.full(20, 7))
    np.testing.assert_allclose(reg.predict([[3], [9]]), [7, 7])


if __name__ == "__main__":
    test_forest()
