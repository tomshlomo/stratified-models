from __future__ import annotations

import cvxpy as cp
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_array

from stratified_models.scalar_function import (
    SparseQuadraticForm,
    SumOfSquares,
    SumOfSquaresOverAffine,
    Zero,
)


def test_zero_and_sum_of_squares_values_and_derivatives() -> None:
    zero = Zero()
    x = jnp.array([1.0, 2.0])

    assert float(zero(x)) == 0.0
    zero_var = cp.Variable(2)
    zero_expr = zero.cvxpy_expression(zero_var)
    zero_var.value = np.array([1.0, 2.0])
    assert zero_expr.value is not None
    assert float(np.asarray(zero_expr.value).item()) == 0.0

    sos = SumOfSquares()
    value = float(sos(x))
    assert np.isclose(value, 2.5)

    grad = sos.gradient()(x)
    hess = sos.hessian()(x)
    assert np.allclose(np.asarray(grad), np.array([1.0, 2.0]))
    assert np.allclose(np.asarray(hess), np.eye(2))


def test_sum_of_squares_over_affine_matches_residual() -> None:
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([1.0, 1.0])
    func = SumOfSquaresOverAffine(a=a, b=b)
    x = jnp.array([1.0, 1.0])

    assert np.allclose(np.asarray(func.residual(x)), np.array([2.0, 6.0]))
    assert np.isclose(float(func(x)), 20.0)

    var = cp.Variable(2)
    expr = func.cvxpy_expression(var)
    var.value = np.array([1.0, 1.0])
    assert expr.value is not None
    assert np.isclose(float(np.asarray(expr.value).item()), 20.0)


def test_sparse_quadratic_form_matches_laplacian_quadratic() -> None:
    laplacian = csr_array(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    func = SparseQuadraticForm(a=laplacian, axis=0, dims=(2, 1))
    x = jnp.array([[1.0], [2.0]])

    assert np.isclose(float(func(x)), 0.5)

    var = cp.Variable((2, 1))
    expr = func.cvxpy_expression(var)
    var.value = np.array([[1.0], [2.0]])
    assert expr.value is not None
    assert np.isclose(float(np.asarray(expr.value).item()), 0.5)
