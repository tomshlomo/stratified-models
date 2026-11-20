from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
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


class Proxable(ScalarFunction):
    @abstractmethod
    def prox(self, v: jax.Array, t: float | jax.Array) -> jax.Array:
        pass


class CachedCholeskyFactorizer:
    def __init__(self, get_matrix: Callable[[], jax.Array]) -> None:
        self._get_matrix = get_matrix
        self._cache: dict[float, tuple[jax.Array, bool]] = {}

    def solve(self, t: float | jax.Array, b: jax.Array) -> jax.Array:
        try:
            t_val = float(t)
        except (TypeError, jax.errors.ConcretizationTypeError):  # JAX Tracer
            m_mat = self._get_matrix()
            i_mat = jnp.eye(m_mat.shape[0])
            matrix = i_mat + t * m_mat
            c_and_lower = jax.scipy.linalg.cho_factor(matrix)
            return jax.scipy.linalg.cho_solve(c_and_lower, b)

        if t_val not in self._cache:
            # M = I + t * A
            # We assume the matrix provided by get_matrix is A
            # and we want to solve (I + t*A)x = b
            m_mat = self._get_matrix()
            i_mat = jnp.eye(m_mat.shape[0])
            matrix = i_mat + t_val * m_mat
            self._cache[t_val] = jax.scipy.linalg.cho_factor(matrix)

        c_and_lower = self._cache[t_val]
        return jax.scipy.linalg.cho_solve(c_and_lower, b)


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
class Zero(QuadraticScalarFunction, CVXPYScalarFunction, Proxable):
    def __call__(self, x: jax.Array) -> jax.Array:  # noqa: ARG002
        return jnp.asarray(0.0)

    def cvxpy_expression(
        self,
        x: cp.Expression,  # noqa: ARG002
    ) -> cp.Expression:
        return cp.Constant(0.0)

    def prox(self, v: jax.Array, t: float | jax.Array) -> jax.Array:  # noqa: ARG002
        return v


class SumOfSquares(QuadraticScalarFunction, CVXPYScalarFunction, Proxable):
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

    def prox(self, v: jax.Array, t: float | jax.Array) -> jax.Array:
        return v / (1 + t)


@attrs.frozen(kw_only=True)
class SumOfSquaresOverAffine(QuadraticScalarFunction, CVXPYScalarFunction, Proxable):
    a: jax.Array
    b: jax.Array
    _factorizer: CachedCholeskyFactorizer = attrs.field(
        init=False, eq=False, repr=False
    )

    def __attrs_post_init__(self) -> None:
        object.__setattr__(
            self, "_factorizer", CachedCholeskyFactorizer(lambda: self.ata)
        )
        # Pre-compute cached properties to avoid side effects (tracer leakage) in JIT
        _ = self.ata
        _ = self.atb

    @cached_property
    def ata(self) -> jax.Array:
        return self.a.T @ self.a

    @cached_property
    def atb(self) -> jax.Array:
        return self.a.T @ self.b

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

    def prox(self, v: jax.Array, t: float | jax.Array) -> jax.Array:
        # (I + t A^T A) x = v + t A^T b
        rhs = v + t * self.atb
        return self._factorizer.solve(t, rhs)


@attrs.frozen(kw_only=True)
class SparseQuadraticForm(QuadraticScalarFunction, CVXPYScalarFunction, Proxable):
    a: sparray
    axis: int
    dims: tuple[int, ...]
    _factorizer: CachedCholeskyFactorizer = attrs.field(
        init=False, eq=False, repr=False
    )

    def __attrs_post_init__(self) -> None:
        object.__setattr__(
            self, "_factorizer", CachedCholeskyFactorizer(lambda: self.a_dense)
        )
        # Pre-compute cached properties to avoid side effects (tracer leakage) in JIT
        _ = self.a_dense
        _ = self.a_jax

    @cached_property
    def a_dense(self) -> jax.Array:
        return jnp.array(self.a.toarray())  # type: ignore[attr-defined]

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

    def prox(self, v: jax.Array, t: float | jax.Array) -> jax.Array:
        # (I + t L) x = v
        v_swapped = jnp.swapaxes(v, self.axis, -1)
        orig_shape_swapped = v_swapped.shape
        v_flat = jnp.reshape(v_swapped, (-1, v_swapped.shape[-1]))

        # Solve (I + t L) X^T = V^T
        sol_t = self._factorizer.solve(t, v_flat.T)
        sol = sol_t.T

        sol_reshaped = jnp.reshape(sol, orig_shape_swapped)
        return jnp.swapaxes(sol_reshaped, self.axis, -1)


@attrs.frozen(kw_only=True)
class SeparableProxableScalarFunction(Proxable):
    items: Sequence[tuple[tuple[int, ...], Proxable]]

    def __call__(self, x: jax.Array) -> jax.Array:
        val = jnp.asarray(0.0)
        for idx, func in self.items:
            val += func(x[idx])
        return val

    def prox(self, v: jax.Array, t: float | jax.Array) -> jax.Array:
        new_v = v
        for idx, func in self.items:
            sub_v = v[idx]
            sub_res = func.prox(sub_v, t)
            new_v = new_v.at[idx].set(sub_res)
        return new_v
