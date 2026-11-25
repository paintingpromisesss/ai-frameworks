import numpy as np

from models.boosting import Boosting


def test_boosting():
    x = np.arange(12)[:, None]
    y = np.where(x.ravel() < 6, 3, 13)
    reg = Boosting('regression', n_estimators=20, max_depth=1, random_state=20).fit(x, y)
    np.testing.assert_allclose(reg.predict([[1], [10]]), [3 + 5 * .9**20, 13 - 5 * .9**20])
    cls = Boosting('classification', n_estimators=10, max_depth=1, random_state=20).fit(x, y)
    np.testing.assert_array_equal(cls.predict(x), y)


if __name__ == "__main__":
    test_boosting()
