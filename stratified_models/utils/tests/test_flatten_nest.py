from stratified_models.utils.flatten_nest import flatten, nest


def test_flatten():
    assert list(flatten(1)) == [1]
    assert list(flatten((1, 2))) == [1, 2]
    assert list(flatten(((1, 2), 3))) == [1, 2, 3]
    assert list(flatten((((1, 2), 3), 4))) == [1, 2, 3, 4]


def test_nest():
    assert 1 == nest([1])
    assert (1, 2) == nest([1, 2])
    assert ((1, 2), 3) == nest([1, 2, 3])
    assert (((1, 2), 3), 4) == nest([1, 2, 3, 4])
