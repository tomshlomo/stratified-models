import numpy as np
import pytest
import scipy.sparse

from stratified_models.simpler.linear_operator import (
    BlockDiagonalLinearOperator,
    FlattenedTensorDot,
    Identity,
    LinearOperator,
    MatrixBasedLinearOperator,
    RepeatedLinearOperator,
    SumOfLinearOperators,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.mark.parametrize("m", [0, 1, 5])
def test_identity_matvec_and_sparse_matrix(m: int) -> None:
    op = Identity(m=m)
    x = _rng().standard_normal(m).astype(float)
    assert op.size() == m
    assert op.shape == (m, m)
    assert np.allclose(op.matvec(x), x, atol=0.0, rtol=0.0)
    assert np.allclose(op.as_sparse_matrix().toarray(), np.eye(m), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("m", [1, 3, 7])
def test_matrix_based_linear_operator_matvec_matches_dense(m: int) -> None:
    rng = _rng()
    a = rng.standard_normal((m, m)).astype(float)
    x = rng.standard_normal(m).astype(float)
    op = MatrixBasedLinearOperator(a=a)

    assert op.size() == m
    assert np.allclose(op.matvec(x), a @ x, atol=0.0, rtol=0.0)

    sparse = op.as_sparse_matrix()
    assert scipy.sparse.issparse(sparse)
    assert np.allclose(sparse.toarray(), a, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("m", "gammas"),
    [
        (3, (0.0, 1.0)),
        (3, (2.0, -0.5)),
        (5, (-1.0, 3.0)),
    ],
)
def test_sum_of_linear_operators_matches_weighted_sum(
    m: int, gammas: tuple[float, ...]
) -> None:
    rng = _rng()
    a1 = rng.standard_normal((m, m)).astype(float)
    a2 = rng.standard_normal((m, m)).astype(float)
    x = rng.standard_normal(m).astype(float)

    op1 = MatrixBasedLinearOperator(a=a1)
    op2 = MatrixBasedLinearOperator(a=a2)
    op = SumOfLinearOperators(components=((op1, gammas[0]), (op2, gammas[1])), m=m)

    expected = gammas[0] * (a1 @ x) + gammas[1] * (a2 @ x)
    assert np.allclose(op.matvec(x), expected, atol=1e-12, rtol=0.0)

    expected_sparse = gammas[0] * scipy.sparse.csr_matrix(a1) + gammas[
        1
    ] * scipy.sparse.csr_matrix(a2)
    assert np.allclose(
        op.as_sparse_matrix().toarray(),
        expected_sparse.toarray(),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("k", "m", "blocks"),
    [
        (3, 2, {0, 2}),
        (4, 3, {1}),
        (2, 5, {0, 1}),
    ],
)
def test_block_diagonal_linear_operator_matvec_and_sparse_matrix(
    k: int, m: int, blocks: set[int]
) -> None:
    rng = _rng()
    ops: dict[int, LinearOperator] = {}
    dense_blocks: list[np.ndarray] = []
    for i in range(k):
        if i in blocks:
            a = rng.standard_normal((m, m)).astype(float)
            ops[i] = MatrixBasedLinearOperator(a=a)
            dense_blocks.append(a)
        else:
            dense_blocks.append(np.zeros((m, m), dtype=float))

    op = BlockDiagonalLinearOperator(blocks=ops, k=k, m=m)
    x = rng.standard_normal(k * m).astype(float)

    expected = np.zeros_like(x)
    for i in range(k):
        s = slice(i * m, (i + 1) * m)
        expected[s] = dense_blocks[i] @ x[s]

    assert op.size() == k * m
    assert np.allclose(op.matvec(x), expected, atol=1e-12, rtol=0.0)

    expected_sparse = scipy.sparse.block_diag(
        [scipy.sparse.csr_matrix(b) for b in dense_blocks]
    ).toarray()
    assert np.allclose(
        op.as_sparse_matrix().toarray(), expected_sparse, atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize(
    ("m", "repetitions"),
    [
        (1, 1),
        (2, 3),
        (4, 2),
    ],
)
def test_repeated_linear_operator_matches_kron(m: int, repetitions: int) -> None:
    rng = _rng()
    a = rng.standard_normal((m, m)).astype(float)
    base = MatrixBasedLinearOperator(a=a)
    op = RepeatedLinearOperator(base_op=base, repetitions=repetitions)

    x = rng.standard_normal(m * repetitions).astype(float)
    dense_from_op = op.as_sparse_matrix().toarray()
    expected_dense = scipy.sparse.kron(
        scipy.sparse.eye(repetitions), scipy.sparse.csr_matrix(a)
    ).toarray()
    assert np.allclose(dense_from_op, expected_dense, atol=1e-12, rtol=0.0)
    assert np.allclose(op.matvec(x), dense_from_op @ x, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    ("dims", "axis"),
    [
        ((2,), 0),
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 2, 3), 2),
    ],
)
def test_flattened_tensor_dot_matvec_matches_sparse_matrix(
    dims: tuple[int, ...], axis: int
) -> None:
    rng = _rng()
    axis_dim = dims[axis]
    a = rng.standard_normal((axis_dim, axis_dim)).astype(float)
    op = FlattenedTensorDot(a=a, axis=axis, dims=dims)

    x = rng.standard_normal(int(np.prod(dims))).astype(float)
    sparse = op.as_sparse_matrix()
    assert np.allclose(op.matvec(x), sparse @ x, atol=1e-12, rtol=0.0)
