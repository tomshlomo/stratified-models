from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Generic, Optional, Protocol, TypeVar, Union

import cvxpy as cp
import numpy as np
from numpy import typing as npt

from stratified_models.linear_operator import FlattenedTensorDot, Identity
from stratified_models.quadratic import ExplicitQuadraticFunction

T = TypeVar("T", contravariant=True)
Array = npt.NDArray[np.float64]


class ScalarFunction(Protocol[T]):
    def __call__(self, x: T) -> float:
        raise NotImplementedError


class QuadraticScalarFunction(ScalarFunction[T], Protocol):
    """
    f(x) = x' q x / 2 - c'x + d/2 for some psd matrix q, vector c, and scalar d
    """

    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        raise NotImplementedError


X = TypeVar("X")


class ProxableScalarFunction(ScalarFunction[X], Protocol):
    def prox(self, v: X, t: float) -> X:
        raise NotImplementedError

    def grad(self, x: X) -> X:
        raise NotImplementedError


@dataclass
class Zero(Generic[X], ProxableScalarFunction[X]):
    def __call__(self, x: X) -> float:
        return 0.0

    def prox(self, v: X, t: float) -> X:
        return v


@dataclass
class SumOfSquares(ProxableScalarFunction[Array], QuadraticScalarFunction[Array]):
    """
    x |-> x'x/2
    x in RefitDataType^m
    """

    shape: int | tuple[int, ...]

    def __call__(self, x: Array) -> float:
        return float((x.ravel() @ x.ravel()) / 2)

    def prox(self, v: Array, t: float) -> Array:
        """
        argmin |x|^2 /2 + |x - v|^2 /2t
        x + (x - v)/t = 0
        tx + x = v
        x = v/(1+t)
        """
        return v / (1 + t)

    def grad(self, x: Array) -> Array:
        return x

    def cvxpy_expression(
        self,
        x: cp.Expression,  # type: ignore[name-defined]
    ) -> cp.Expression:  # type: ignore[name-defined]
        return cp.sum_squares(x)  # type: ignore[attr-defined]

    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        m = int(np.prod(self.shape))
        return ExplicitQuadraticFunction(
            q=Identity(m),
            c=np.zeros(m),
            d=0.0,
        )


# @dataclass
# class Affine(ProxableScalarFunction[Array], QuadraticScalarFunction[Array]):
#     """
#     x |-> c'x + d
#     x in RefitDataType^m
#     """
#     c: Array
#     d: float
#
#     def __call__(self, x: Array) -> float:
#         return float((x.ravel() @ self.c.ravel())) + self.d
#
#     def prox(self, v: Array, t: float) -> Array:
#         """
#         argmin c'x + d + 1/2t |x-v|^2
#         c + 1/t(x - v) = 0
#         x = v - tc
#         """
#         return v - t * self.c
#
#     def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
#         m = int(np.prod(self.shape))
#         return ExplicitQuadraticFunction(
#             q=Identity(m),
#             c=np.zeros(m),
#             d=0.0,
#         )


def soft_threshold(x: Array, thresh: float) -> Array:
    return np.clip(x - thresh, 0.0, None) - np.clip(-x - thresh, 0.0, None)


class L1:
    def __call__(self, x: Array) -> float:
        return float(np.sum(np.abs(x)))

    def prox(self, v: Array, t: float) -> Array:
        """
        argmin |x| + 1/2t |x - v|^2
        sign(x) + (x - v)/t = 0
        assume x is positive:
        t + x - v = 0
        x = v - t
        """
        return soft_threshold(v, t)


@dataclass
class NonNegativeIndicator:
    def __call__(self, x: Array) -> float:
        return float("inf") if (x < 0).any() else 0.0

    def prox(self, v: Array, t: float) -> Array:
        return np.clip(v, 0.0, None)


EinsumPath = list[Union[str, tuple[int, ...]]]


@dataclass
class _TensorQuadFormProxCache:
    einsum_subscripts: str
    einsum_path: EinsumPath
    u: Array
    d: Array


@dataclass
class _TensorQuadFormCallCache:
    einsum_subscripts: str
    einsum_path: EinsumPath


@dataclass
class TensorQuadForm(QuadraticScalarFunction[Array], ProxableScalarFunction[Array]):
    axis: int
    dims: tuple[int, ...]
    a: npt.NDArray[
        np.float64
    ]  # todo: could also be a pydata.sparse array, which also supports tensordot
    _call_cache: Optional[_TensorQuadFormCallCache] = None
    _prox_cache: Optional[_TensorQuadFormProxCache] = None

    def _set_call_einsum_args_cache(self) -> None:
        if self._call_cache:
            return
        all_letters = string.ascii_letters
        summation_index1 = all_letters[-1]
        summation_index2 = all_letters[self.axis]
        x2_subs = all_letters[: len(self.dims)]
        a_subs = summation_index1 + summation_index2
        x1_subs = x2_subs.replace(summation_index2, summation_index1, 1)
        subscripts = f"{x1_subs},{a_subs},{x2_subs}"

        x = np.empty(self.dims, dtype=self.a.dtype)
        path, path_str = np.einsum_path(subscripts, x, self.a, x, optimize="optimal")
        self._call_cache = _TensorQuadFormCallCache(
            einsum_subscripts=subscripts, einsum_path=path
        )

    def _set_prox_cache(self) -> None:
        if self._prox_cache:
            return
        einsum_subscripts, einsum_path = self._prox_einsum_args()
        d, u = np.linalg.eigh(self.a)
        self._prox_cache = _TensorQuadFormProxCache(
            einsum_path=einsum_path,
            einsum_subscripts=einsum_subscripts,
            d=d,
            u=u,
        )

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
        path, path_str = np.einsum_path(subscripts, u, w, u, x, optimize="optimal")
        return subscripts, path

    def __call__(self, x: Array) -> float:
        self._set_call_einsum_args_cache()
        subscripts = self._call_cache.einsum_subscripts
        path = self._call_cache.einsum_path
        out = np.einsum(subscripts, x, self.a, x, optimize=path)
        return float(out) / 2

    def grad(self, x: Array) -> Array:
        all_letters = string.ascii_letters
        out_index = all_letters[-1]
        summation_index = all_letters[self.axis]
        x_subs = all_letters[: len(self.dims)]
        a_subs = out_index + summation_index
        out_subs = x_subs.replace(summation_index, out_index)
        subscripts = f"{a_subs},{x_subs}->{out_subs}"
        out = np.einsum(
            subscripts,
            self.a,
            x,
        )
        return out

    def prox(self, v: Array, t: float) -> Array:
        """
        argmin x' a x / 2 + |x - v|^2 / 2t
        t a x + (x - v) = 0
        (ta + I) x = v
        x = (ta + I)^-1 v

        let a = udu' be the eigen decomposition
        so:
        x = uwu'v
        where w = (td + I)^-1
        """
        self._set_prox_cache()
        u = self._prox_cache.u
        d = self._prox_cache.d
        einsum_subscripts = self._prox_cache.einsum_subscripts
        einsum_path = self._prox_cache.einsum_path

        w = 1 / (t * d + 1)
        return np.einsum(einsum_subscripts, u, w, u, v, optimize=einsum_path)

    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        return ExplicitQuadraticFunction.quadratic_form(
            q=FlattenedTensorDot(
                a=self.a,
                axis=self.axis,
                dims=self.dims,
            )
        )

    def cvxpy_expression(
        self,
        x: cp.Expression,  # type: ignore[name-defined]
    ) -> cp.Expression:  # type: ignore[name-defined]
        q = self.to_explicit_quadratic().q.as_sparse_matrix()
        return cp.quad_form(x, q, assume_PSD=True) / 2  # type: ignore[attr-defined]
