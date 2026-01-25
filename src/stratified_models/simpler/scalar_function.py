from __future__ import annotations

from abc import ABC, abstractmethod

import attrs
import cvxpy as cp
import jax
import jax.numpy as jnp
from scipy.sparse import sparray


class ScalarFunction(ABC):
    @abstractmethod
    def __call__(self, x: jax.Array) -> float:
        pass


class CVXPYScalarFunction(ScalarFunction, ABC):
    @abstractmethod
    def cvxpy_expression(
        self,
        x: cp.Expression,
    ) -> cp.Expression:
        pass


@attrs.frozen(kw_only=True)
class Zero(ScalarFunction):
    def __call__(self, x: jax.Array) -> float:  # noqa: ARG002
        return 0.0

    def cvxpy_expression(
        self,
        x: cp.Expression,  # noqa: ARG002
    ) -> cp.Expression:
        return cp.Constant(0.0)


@attrs.frozen(kw_only=True)
class SumOfSquares(CVXPYScalarFunction):
    """x |-> x'x/2
    x in RefitDataType^m.
    """

    shape: int | tuple[int, ...]

    def __call__(self, x: jax.Array) -> float:
        x_flat = jnp.ravel(x)
        return float((x_flat @ x_flat) / 2)

    def cvxpy_expression(
        self,
        x: cp.Expression,
    ) -> cp.Expression:
        return cp.sum_squares(x) / 2


@attrs.frozen(kw_only=True)
class SumOfSquaresOverAffine(CVXPYScalarFunction):
    a: jax.Array
    b: jax.Array

    def __call__(self, x: jax.Array) -> float:
        residual = self.residual(x)
        residual_flat = jnp.ravel(residual)
        return float((residual_flat @ residual_flat) / 2)

    def residual(self, x: jax.Array) -> jax.Array:
        return self.a @ x - self.b

    def cvxpy_expression(
        self,
        x: cp.Expression,
    ) -> cp.Expression:
        return cp.sum_squares(self.a @ x - self.b) / 2


@attrs.frozen(kw_only=True)
class SparseQuadraticForm(CVXPYScalarFunction):
    a: sparray
    axis: int
    dims: tuple[int, ...]

    def __call__(self, x: jax.Array) -> float:
        assert x.shape == self.dims
        # `self.a` is a SciPy sparse matrix (CPU-only), so we evaluate this
        # expression on host arrays even if `x` is a JAX array.
        x_arr = jnp.swapaxes(x, self.axis, -1)
        x_arr = jnp.reshape(x_arr, (-1, x_arr.shape[-1]))
        x_np = jax.device_get(x_arr)
        y_np = x_np @ self.a
        return float(x_np.ravel() @ y_np.ravel()) / 2

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
