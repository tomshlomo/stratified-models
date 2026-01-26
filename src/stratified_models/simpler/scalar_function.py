from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property

import attrs
import cvxpy as cp
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from scipy.sparse import sparray


class ScalarFunction(ABC):
    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        pass


class CVXPYScalarFunction(ScalarFunction, ABC):
    @abstractmethod
    def cvxpy_expression(
        self,
        x: cp.Expression,
    ) -> cp.Expression:
        pass


class QuadraticScalarFunction(ScalarFunction, ABC):
    def gradient(self) -> Callable[[jax.Array], jax.Array]:
        return jax.grad(self.__call__)

    def hessian(self) -> Callable[[jax.Array], jax.Array]:
        return jax.hessian(self.__call__)


@attrs.frozen(kw_only=True)
class Zero(QuadraticScalarFunction, CVXPYScalarFunction):
    def __call__(self, x: jax.Array) -> jax.Array:  # noqa: ARG002
        return jnp.asarray(0.0)

    def cvxpy_expression(
        self,
        x: cp.Expression,  # noqa: ARG002
    ) -> cp.Expression:
        return cp.Constant(0.0)


class SumOfSquares(QuadraticScalarFunction, CVXPYScalarFunction):
    """x |-> x'x/2
    x in RefitDataType^m.
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        x_flat = jnp.ravel(x)
        return (x_flat @ x_flat) / 2

    def cvxpy_expression(
        self,
        x: cp.Expression,
    ) -> cp.Expression:
        return cp.sum_squares(x) / 2


@attrs.frozen(kw_only=True)
class SumOfSquaresOverAffine(QuadraticScalarFunction, CVXPYScalarFunction):
    a: jax.Array
    b: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = self.residual(x)
        residual_flat = jnp.ravel(residual)
        return (residual_flat @ residual_flat) / 2

    def residual(self, x: jax.Array) -> jax.Array:
        return self.a @ x - self.b

    def cvxpy_expression(
        self,
        x: cp.Expression,
    ) -> cp.Expression:
        return cp.sum_squares(self.a @ x - self.b) / 2


@attrs.frozen(kw_only=True)
class SparseQuadraticForm(QuadraticScalarFunction, CVXPYScalarFunction):
    a: sparray
    axis: int
    dims: tuple[int, ...]

    @cached_property
    def a_jax(self) -> BCOO:
        # Convert the (CPU) SciPy Laplacian to a JAX sparse matrix once.
        return BCOO.from_scipy_sparse(self.a)

    def __call__(self, x: jax.Array) -> jax.Array:
        assert x.shape == self.dims
        x_arr = jnp.swapaxes(x, self.axis, -1)
        x_arr = jnp.reshape(x_arr, (-1, x_arr.shape[-1]))
        # Quadratic form: sum_i x_i^T L x_i / 2
        y = (self.a_jax @ x_arr.T).T
        return jnp.sum(x_arr * y) / 2

    def cvxpy_expression(
        self,
        x: cp.Expression,
    ) -> cp.Expression:
        assert x.shape == self.dims
        x = cp.swapaxes(x, self.axis, -1)
        x = cp.reshape(x, (-1, x.shape[-1]))
        out = cp.Constant(0.0)
        for xx in x:
            out += cp.quad_form(xx, self.a, assume_PSD=True)
        return out / 2
