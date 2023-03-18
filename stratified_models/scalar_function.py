from __future__ import annotations

import string
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, Protocol, TypeVar

import cvxpy as cp
import numpy as np
from numpy import typing as npt

from stratified_models.linear_operator import FlattenedTensorDot, Identity
from stratified_models.quadratic import ExplicitQuadraticFunction

T = TypeVar("T")
Vector = npt.NDArray[np.float64]
DenseMatrix = npt.NDArray[np.float64]


class ScalarFunction(Protocol[T]):
    def __call__(self, x: T) -> float:
        pass


# K = TypeVar('K', bound=Hashable)
#
#
# class Mappable(Protocol[K, T]):
#     def __getitem__(self, k: K) -> T:
#         pass
#
#
# class SeparableFunction(Mappable[K, T]):
#     f: dict[K, ScalarFunction]
#
#     def __call__(self, x: Mappable[K, T]) -> float:
#         return sum(f(x[k]) for k, f in self.f.items())
#
#     def prox(self, v: Mappable[K, T], t: float) -> Mappable[K, T]:
#         for k, f in self.f.items():
#             x[k] = f.prox(v[k], t)


@dataclass
class Zero(Generic[T]):
    def __call__(self, x: T) -> float:
        return 0.0

    def prox(self, v: T, t: float) -> T:
        return v


class QuadraticScalarFunction(ScalarFunction[T]):
    """
    f(x) = x' q x / 2 - c'x + d/2 for some psd matrix q, vector c, and scalar d
    """

    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        pass

    # def dense_matrix(self) -> DenseMatrix:
    #     pass


class ProxableScalarFunction(ScalarFunction[T]):
    def prox(self, v: T, t: float) -> T:
        pass


@dataclass
class SumOfSquares:
    """
    x |-> x'x/2
    x in R^m
    """

    shape: int | tuple[int, ...]

    def __call__(self, x: Vector) -> float:
        return (x.ravel() @ x.ravel()) / 2

    def prox(self, v: Vector, t: float) -> float:
        """
        argmin |x|^2 /2 + |x - v|^2 /2t
        x + (x - v)/t = 0
        tx + x = v
        x = v/(1+t)
        """
        return v / (1 + t)

    # def matrix(self) -> DenseMatrix:
    #     return np.eye(self.m)

    def cvxpy_expression(self, x: cp.Expression) -> cp.Expression:
        return cp.sum_squares(x)

    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        m = int(np.prod(self.shape))
        return ExplicitQuadraticFunction(
            q=Identity(m),
            c=np.zeros(m),
            d=0.0,
        )


# @dataclass
# class QuadForm:
#     """
#     x' q x / 2
#     """
#
#     q: LinearOperator
#
#     def __call__(self, x: Vector) -> float:
#         return (x @ (self.q.matvec(x))) / 2
#
#     def prox(self, v: Vector, t: float) -> Vector:
#         """
#         argmin x' q x / 2 + |x - v|^2 / 2t
#         t q x + (x - v) = 0
#         (tq + I) x = v
#         x = (tq + I)^-1 v
#         """
#         # todo: factorization caching
#         raise NotImplementedError
#         return np.linalg.solve(t * self.q + np.eye(self.q.shape[0]), v)
#
#     def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
#         return ExplicitQuadraticFunction.quadratic_form(self.q)
#
#     def cvxpy_expression(self, x: cp.Expression) -> cp.Expression:
#         return cp.quad_form(x, self.q.as_sparse_matrix(), assume_PSD=True)


# @dataclass
# # todo: is this class even used?
# # todo: implement a matrix method (kron(I, q) or kron(q, I) I always get confused)
# class MatrixQuadForm:
#     """
#     tr(mat(x)' q max(x)) / 2
#     """
#
#     q: DenseMatrix
#
#     def __call__(self, x: Vector) -> float:
#         # todo: einsum
#         x = x.reshape((self.q.shape[0], -1))
#         qx = self.q @ x
#         return np.sum(x * qx) / 2
#
#     def prox(self, v: Vector, t: float) -> Vector:
#         v = v.reshape((self.q.shape[0], -1))
#         return np.linalg.solve(self.q * t + np.eye(self.q.shape[0]), v).ravel()


def soft_threshold(
    x: npt.NDArray[np.float64], thresh: float
) -> npt.NDArray[np.float64]:
    return np.clip(x - thresh, 0.0, None) - np.clip(-x - thresh, 0.0, None)


class L1:
    def __call__(self, x: Vector) -> float:
        return float(np.sum(np.abs(x)))

    def prox(self, v: Vector, t: float) -> Vector:
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
    def __call__(self, x: Vector) -> float:
        return float("inf") if (x < 0).any() else 0.0

    def prox(self, v: Vector, t: float) -> Vector:
        return np.clip(v, 0.0, None)


@dataclass
class TensorQuadForm:
    axis: int
    dims: tuple[int, ...]
    a: npt.NDArray[
        np.float64
    ]  # todo: could also be a pydata.sparse array, which also supports tensordot

    @cached_property
    def call_einsum_args(self):
        all_letters = string.ascii_letters
        summation_index1 = all_letters[-1]
        summation_index2 = all_letters[self.axis]
        x2_subs = all_letters[: len(self.dims)]
        a_subs = summation_index1 + summation_index2
        x1_subs = x2_subs.replace(summation_index2, summation_index1, 1)
        subscripts = f"{x1_subs},{a_subs},{x2_subs}"

        x = np.empty(self.dims, dtype=self.a.dtype)
        path, path_str = np.einsum_path(subscripts, x, self.a, x, optimize="optimal")
        return subscripts, path

    @cached_property
    def prox_einsum_args(self):
        all_letters = string.ascii_letters
        summation_index1 = all_letters[-1]
        summation_index2 = all_letters[self.axis]
        x2_subs = all_letters[: len(self.dims)]
        a_subs = summation_index1 + summation_index2
        out_subs = x2_subs.replace(summation_index2, summation_index1)
        subscripts = f"{a_subs},{x2_subs}->{out_subs}"

        x = np.empty(self.dims, dtype=self.a.dtype)
        path, path_str = np.einsum_path(subscripts, self.a, x, optimize="optimal")
        return subscripts, path

    def __call__(self, x: npt.NDArray[np.float64]) -> float:
        subscripts, path = self.call_einsum_args
        out = np.einsum(subscripts, x, self.a, x, optimize=path)
        return float(out) / 2
        # ax = np.tensordot(self.a, x, axes=(1, self.axis))
        # return (x.ravel() @ ax.ravel()) / 2

    def prox(self, v: npt.NDArray[np.float64], t: float) -> npt.NDArray[np.float64]:
        """
        argmin x' a x / 2 + |x - v|^2 / 2t
        t a x + (x - v) = 0
        (ta + I) x = v
        x = (ta + I)^-1 v
        :param v:
        :param t:
        :return:
        """
        p = np.linalg.inv(t * self.a + np.eye(self.a.shape[0]))
        subscripts, path = self.prox_einsum_args
        x = np.einsum(subscripts, p, v, optimize=path)
        return x

    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        return ExplicitQuadraticFunction.quadratic_form(
            q=FlattenedTensorDot(
                a=self.a,
                axis=self.axis,
                dims=self.dims,
            )
        )

    def cvxpy_expression(self, x: cp.Expression) -> cp.Expression:
        q = self.to_explicit_quadratic().q.as_sparse_matrix()
        return cp.quad_form(x, q, assume_PSD=True) / 2
