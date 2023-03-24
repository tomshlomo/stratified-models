import numpy as np

from stratified_models.linear_operator import (
    BlockDiagonalLinearOperator,
    MatrixBasedLinearOperator,
    RepeatedLinearOperator,
)


def test_matrix_based_linear_operator() -> None:
    m = 5
    a = np.arange(m**2).reshape((m, m))
    op = MatrixBasedLinearOperator(a)
    assert op.shape == (m, m)
    assert op.size() == m
    scipy_op = op.to_scipy_linear_operator()
    mat = scipy_op.matmat(np.eye(m))
    assert np.array_equal(a, mat)


def test_repeated_linear_operator() -> None:
    m = 5
    k = 3
    a = np.arange(m**2).reshape((m, m))
    base_op = MatrixBasedLinearOperator(a)
    op = RepeatedLinearOperator(base_op, k)
    assert op.size() == m * k


def test_block_diagonal_linear_operator() -> None:
    m = 5
    k = 3
    blocks = {
        0: np.arange(m**2).reshape((m, m)),
        2: np.eye(m),
    }
    op = BlockDiagonalLinearOperator(blocks=blocks, m=m, k=k)
    assert op.size() == m * k
