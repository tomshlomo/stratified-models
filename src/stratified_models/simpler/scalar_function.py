from __future__ import annotations

import string
from abc import ABC, abstractmethod
from functools import cached_property

import attrs
import cvxpy as cp
import numpy as np
from numpy import typing as npt

from stratified_models.simpler.linear_operator import (
    FlattenedTensorDot,
    Identity,
    MatrixBasedLinearOperator,
)
from stratified_models.simpler.quadratic import ExplicitQuadraticFunction

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


class ScalarFunction(ABC):
    @abstractmethod
    def __call__(self, x: Array) -> float:
        pass


class QuadraticScalarFunction(ScalarFunction, ABC):
    """f(x) = x' q x / 2 - c'x + d/2 for some psd matrix q, vector c, and scalar d."""

    @abstractmethod
    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        pass


class ProxableScalarFunction(ScalarFunction, ABC):
    @abstractmethod
    def prox(self, v: Array, t: float) -> Array:
        pass


class CVXPYScalarFunction(ScalarFunction, ABC):
    @abstractmethod
    def cvxpy_expression(
        self,
        x: cp.Expression,
    ) -> cp.Expression:
        pass


@attrs.frozen(kw_only=True)
class Zero(ProxableScalarFunction, QuadraticScalarFunction):
    def __call__(self, x: Array) -> float:  # noqa: ARG002
        return 0.0

    def prox(self, v: Array, t: float) -> Array:  # noqa: ARG002
        return v

    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        return ExplicitQuadraticFunction(q=Identity(0), c=np.zeros(0), d=0.0)

    def cvxpy_expression(
        self,
        x: cp.Expression,  # noqa: ARG002
    ) -> cp.Expression:
        return cp.Constant(0.0)


@attrs.frozen(kw_only=True)
class SumOfSquares(
    ProxableScalarFunction, QuadraticScalarFunction, CVXPYScalarFunction
):
    """x |-> x'x/2
    x in RefitDataType^m.
    """

    shape: int | tuple[int, ...]

    def __call__(self, x: Array) -> float:
        return float((x.ravel() @ x.ravel()) / 2)

    def prox(self, v: Array, t: float) -> Array:
        """Argmin |x|^2 /2 + |x - v|^2 /2t
        x + (x - v)/t = 0
        tx + x = v
        x = v/(1+t).
        """
        return v / (1 + t)

    def cvxpy_expression(
        self,
        x: cp.Expression,
    ) -> cp.Expression:
        return cp.sum_squares(x) / 2

    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        m = int(np.prod(self.shape))
        return ExplicitQuadraticFunction(
            q=Identity(m),
            c=np.zeros(m),
            d=0.0,
        )


def soft_threshold(x: Array, thresh: float) -> Array:
    return np.clip(x - thresh, 0.0, None) - np.clip(-x - thresh, 0.0, None)


class L1(ProxableScalarFunction, CVXPYScalarFunction):
    def __call__(self, x: Array) -> float:
        return float(np.sum(np.abs(x)))

    def prox(self, v: Array, t: float) -> Array:
        """Argmin |x| + 1/2t |x - v|^2
        sign(x) + (x - v)/t = 0
        assume x is positive:
        t + x - v = 0
        x = v - t.
        """
        if t == 0.0:
            return v
        return soft_threshold(v, t)


@attrs.frozen(kw_only=True)
class NonNegativeIndicator(ProxableScalarFunction, CVXPYScalarFunction):
    def __call__(self, x: Array) -> float:
        return float("inf") if (x < 0).any() else 0.0

    def prox(self, v: Array, t: float) -> Array:  # noqa: ARG002
        return np.clip(v, 0.0, None)


@attrs.frozen(kw_only=True)
class SumOfSquaresOverAffine(
    ProxableScalarFunction, QuadraticScalarFunction, CVXPYScalarFunction
):
    a: Array
    b: Array

    @cached_property
    def _ata(self) -> Array:
        return self.a.T @ self.a

    @cached_property
    def _atb(self) -> Array:
        return self.a.T @ self.b

    def __call__(self, x: Array) -> float:
        residual = self.residual(x)
        return float(residual @ residual / 2)

    def residual(self, x: Array) -> Array:
        return self.a @ x - self.b

    def prox(self, v: Array, t: float) -> Array:
        """Argmin_x |Ax - b|^2 / 2 + |x - v|^2 / (2t).

        This is a ridge-regularized least squares system:
            (t AᵀA + I) x = v + t Aᵀb
        """
        if t == 0.0:
            return v
        m = self.a.shape[1]
        lhs = t * self._ata + np.eye(m)
        rhs = v + t * self._atb
        return np.linalg.solve(lhs, rhs)

    def cvxpy_expression(
        self,
        x: cp.Expression,
    ) -> cp.Expression:
        return cp.sum_squares(self.a @ x - self.b) / 2

    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        """|Ax - b|^2 / 2 = xᵀ(AᵀA)x / 2 - (Aᵀb)ᵀ x + (bᵀb)/2."""
        return ExplicitQuadraticFunction(
            q=MatrixBasedLinearOperator(self._ata),
            c=-self._atb,
            d=float((self.b @ self.b) / 2),
        )


EinsumPath = list[str | tuple[int, ...]]


@attrs.frozen(kw_only=True)
class TensorQuadForm(
    QuadraticScalarFunction, ProxableScalarFunction, CVXPYScalarFunction
):
    axis: int
    dims: tuple[int, ...]
    a: npt.NDArray[
        np.float64
    ]  # TODO: could also be a pydata.sparse array, which also supports tensordot

    @cached_property
    def _call_einsum_args(self) -> tuple[str, EinsumPath]:
        all_letters = string.ascii_letters
        summation_index1 = all_letters[-1]
        summation_index2 = all_letters[self.axis]
        x2_subs = all_letters[: len(self.dims)]
        a_subs = summation_index1 + summation_index2
        x1_subs = x2_subs.replace(summation_index2, summation_index1, 1)
        subscripts = f"{x1_subs},{a_subs},{x2_subs}"

        x = np.empty(self.dims, dtype=self.a.dtype)
        path, _path_str = np.einsum_path(subscripts, x, self.a, x, optimize="optimal")
        return subscripts, path

    def __call__(self, x: Array) -> float:
        subscripts, path = self._call_einsum_args
        out = np.einsum(subscripts, x, self.a, x, optimize=path)
        return float(out) / 2

    @cached_property
    def _prox_einsum_args(self) -> tuple[str, EinsumPath]:
        all_letters = string.ascii_letters
        summation_index1 = all_letters[-1]
        summation_index2 = all_letters[self.axis]
        eig_index = all_letters[-2]
        x2_subs = all_letters[: len(self.dims)]
        a_subs = (
            "nm,m,km".replace("n", summation_index1)
            .replace("k", summation_index2)
            .replace("m", eig_index)
        )
        out_subs = x2_subs.replace(summation_index2, summation_index1)
        subscripts = f"{a_subs},{x2_subs}->{out_subs}"

        x = np.empty(self.dims, dtype=self.a.dtype)
        u = np.empty(self.a.shape, dtype=self.a.dtype)
        w = np.empty(self.a.shape[0], dtype=self.a.dtype)
        path, _path_str = np.einsum_path(subscripts, u, w, u, x, optimize="optimal")
        return subscripts, path

    def prox(self, v: Array, t: float) -> Array:
        """Argmin x' a x / 2 + |x - v|^2 / 2t
        t a x + (x - v) = 0
        (ta + I) x = v
        x = (ta + I)^-1 v.

        let a = udu' be the eigen decomposition
        so:
        x = uwu'v
        where w = (td + I)^-1
        """
        if t == 0.0:
            return v
        subscripts, path = self._prox_einsum_args
        d, u = np.linalg.eigh(self.a)
        w = 1 / (t * d + 1)
        return np.einsum(
            subscripts,
            u,
            w,
            u,
            v,
            optimize=path,
        )

    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        return ExplicitQuadraticFunction.quadratic_form(
            q=FlattenedTensorDot(
                a=self.a,
                axis=self.axis,
                dims=self.dims,
            ),
        )

    def cvxpy_expression(
        self,
        x: cp.Expression,
    ) -> cp.Expression:
        q = self.to_explicit_quadratic().q.as_sparse_matrix()
        expression = cp.quad_form(
            # `ThetaShape.dims` is `(m, *graph_sizes)` while the CVXPY variable
            # is shaped `(num_nodes, m)`. Flattening in Fortran order makes the
            # vectorization consistent with the Kronecker structure implied by
            # `dims` (feature-major blocks).
            x.flatten(order="F"),
            q,
            assume_PSD=True,
        )
        return expression / 2
