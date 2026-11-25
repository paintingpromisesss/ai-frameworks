import numpy as np

from models.tree import Tree


def test_tree():
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [1, 2, 2, 1]
    cls = Tree('classification', max_depth=2).fit(x, y)
    np.testing.assert_array_equal(cls.predict(x), y)
    reg = Tree('regression', min_samples_leaf=2).fit([[0], [1], [2], [3]], [1, 3, 9, 11])
    np.testing.assert_allclose(reg.predict([[0.8], [2.2]]), [2, 10])


if __name__ == "__main__":
    test_tree()
