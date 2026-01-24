import numpy as np
import pytest
import scipy.sparse

from stratified_models.simpler.linear_operator import (
    BlockDiagonalLinearOperator,
    Identity,
    MatrixBasedLinearOperator,
    SumOfLinearOperators,
)
from stratified_models.simpler.quadratic import ExplicitQuadraticFunction


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.mark.parametrize("m", [1, 3, 8])
def test_explicit_quadratic_call_matches_definition(m: int) -> None:
    rng = _rng()
    q_dense = rng.standard_normal((m, m)).astype(float)
    c = rng.standard_normal(m).astype(float)
    d = float(rng.standard_normal())
    x = rng.standard_normal(m).astype(float)

    q = MatrixBasedLinearOperator(a=q_dense)
    f = ExplicitQuadraticFunction(q=q, c=c, d=d)

    expected = float((x @ (q_dense @ x)) / 2 + x @ c + d)
    assert abs(f(x) - expected) <= 1e-12


@pytest.mark.parametrize("m", [1, 4])
def test_quadratic_form_has_zero_linear_and_constant_terms(m: int) -> None:
    rng = _rng()
    q_dense = rng.standard_normal((m, m)).astype(float)
    q = MatrixBasedLinearOperator(a=q_dense)
    f = ExplicitQuadraticFunction.quadratic_form(q=q)

    assert np.allclose(f.c, np.zeros(m), atol=0.0, rtol=0.0)
    assert f.d == 0.0

    x = rng.standard_normal(m).astype(float)
    expected = float((x @ (q_dense @ x)) / 2)
    assert abs(f(x) - expected) <= 1e-12


@pytest.mark.parametrize(
    ("m", "gammas"),
    [
        (3, (0.5, 2.0)),
        (5, (-1.0, 0.25)),
    ],
)
def test_quadratic_sum_matches_weighted_sum(
    m: int, gammas: tuple[float, float]
) -> None:
    rng = _rng()
    q1 = rng.standard_normal((m, m)).astype(float)
    q2 = rng.standard_normal((m, m)).astype(float)
    c1 = rng.standard_normal(m).astype(float)
    c2 = rng.standard_normal(m).astype(float)
    d1 = float(rng.standard_normal())
    d2 = float(rng.standard_normal())

    f1 = ExplicitQuadraticFunction(q=MatrixBasedLinearOperator(a=q1), c=c1, d=d1)
    f2 = ExplicitQuadraticFunction(q=MatrixBasedLinearOperator(a=q2), c=c2, d=d2)

    f = ExplicitQuadraticFunction.sum(
        m=m, components=((f1, gammas[0]), (f2, gammas[1]))
    )

    assert isinstance(f.q, SumOfLinearOperators)
    x = rng.standard_normal(m).astype(float)
    expected = gammas[0] * f1(x) + gammas[1] * f2(x)
    assert abs(f(x) - expected) <= 1e-12

    expected_q = gammas[0] * scipy.sparse.csr_matrix(q1) + gammas[
        1
    ] * scipy.sparse.csr_matrix(q2)
    assert np.allclose(
        f.q.as_sparse_matrix().toarray(), expected_q.toarray(), atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize(
    ("k", "m", "present_blocks"),
    [
        (3, 2, {0, 2}),
        (4, 3, {1}),
    ],
)
def test_quadratic_concat_builds_block_diagonal_operator(
    k: int, m: int, present_blocks: set[int]
) -> None:
    rng = _rng()
    components: dict[int, ExplicitQuadraticFunction] = {}
    dense_blocks: list[np.ndarray] = []
    c = np.zeros((k, m), dtype=float)
    d_sum = 0.0
    for i in range(k):
        if i in present_blocks:
            qi = rng.standard_normal((m, m)).astype(float)
            ci = rng.standard_normal(m).astype(float)
            di = float(rng.standard_normal())
            components[i] = ExplicitQuadraticFunction(
                q=MatrixBasedLinearOperator(a=qi),
                c=ci,
                d=di,
            )
            dense_blocks.append(qi)
            c[i] = ci
            d_sum += di
        else:
            dense_blocks.append(np.zeros((m, m), dtype=float))

    f = ExplicitQuadraticFunction.concat(k=k, m=m, components=components)
    assert isinstance(f.q, BlockDiagonalLinearOperator)
    assert np.allclose(f.c, c.ravel(), atol=0.0, rtol=0.0)
    assert abs(f.d - d_sum) <= 1e-12

    q_dense = scipy.sparse.block_diag(
        [scipy.sparse.csr_matrix(b) for b in dense_blocks]
    ).toarray()
    assert np.allclose(f.q.as_sparse_matrix().toarray(), q_dense, atol=1e-12, rtol=0.0)

    x = rng.standard_normal(k * m).astype(float)
    expected = float((x @ (q_dense @ x)) / 2 + x @ f.c + f.d)
    assert abs(f(x) - expected) <= 1e-12


@pytest.mark.parametrize(
    ("m", "repetitions"),
    [
        (1, 4),
        (3, 2),
    ],
)
def test_quadratic_repeat_matches_kron_and_repeated_terms(
    m: int, repetitions: int
) -> None:
    rng = _rng()
    q_dense = rng.standard_normal((m, m)).astype(float)
    c = rng.standard_normal(m).astype(float)
    d = float(rng.standard_normal())
    f = ExplicitQuadraticFunction(q=MatrixBasedLinearOperator(a=q_dense), c=c, d=d)

    fr = f.repeat(repetitions=repetitions)
    expected_q = scipy.sparse.kron(
        scipy.sparse.eye(repetitions),
        scipy.sparse.csr_matrix(q_dense),
    ).toarray()
    assert np.allclose(
        fr.q.as_sparse_matrix().toarray(), expected_q, atol=1e-12, rtol=0.0
    )
    assert np.allclose(fr.c, c.repeat(repetitions), atol=0.0, rtol=0.0)
    assert abs(fr.d - d * repetitions) <= 1e-12

    x = rng.standard_normal(m * repetitions).astype(float)
    expected = float((x @ (expected_q @ x)) / 2 + x @ fr.c + fr.d)
    assert abs(fr(x) - expected) <= 1e-12


@pytest.mark.parametrize("m", [0, 5])
def test_quadratic_form_with_identity_matches_half_norm_squared(m: int) -> None:
    rng = _rng()
    q = Identity(m=m)
    f = ExplicitQuadraticFunction.quadratic_form(q=q)
    x = rng.standard_normal(m).astype(float)
    expected = float((x @ x) / 2)
    assert abs(f(x) - expected) <= 1e-12
