from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, Union

import cvxpy as cp
import numpy as np
import scipy.sparse

from stratified_models.linear_operator import MatrixBasedLinearOperator
from stratified_models.quadratic import ExplicitQuadraticFunction
from stratified_models.scalar_function import (
    Array,
    ProxableScalarFunction,
    QuadraticScalarFunction,
    ScalarFunction,
)

L = TypeVar("L", bound=ScalarFunction[Array], covariant=True)


class LossFactory(Generic[L]):
    @abstractmethod
    def build_loss_function(self, x: Array, y: Array) -> L:
        raise NotImplementedError


EinsumPath = list[Union[str, tuple[int, ...]]]


@dataclass
class SumOfSquaresProxCache:
    atb: Array
    u: Array
    d: Array
    einsum_path: EinsumPath


def _to_numpy_array(x: Union[Array, scipy.sparse.spmatrix]) -> Array:
    return x.toarray() if isinstance(x, scipy.sparse.spmatrix) else x


@dataclass
class SumOfSquaresLoss(QuadraticScalarFunction[Array], ProxableScalarFunction[Array]):
    a: Union[Array, scipy.sparse.spmatrix]
    b: Array
    _prox_cache: Optional[SumOfSquaresProxCache] = None

    def __call__(self, x: Array) -> float:
        y = self.a @ x
        r = y - self.b
        return float((r @ r) / 2)

    def _set_prox_cache(self) -> None:
        atb = self.a.T @ self.b
        if self.is_tall:
            d, u = np.linalg.eigh(_to_numpy_array(self.a.T @ self.a))
        else:
            d, u = np.linalg.eigh(_to_numpy_array(self.a @ self.a.T))
            u = self.a.T @ u
        path, path_str = np.einsum_path(
            "nm,m,km,k->n", u, d, u, np.zeros(u.shape[0]), optimize="optimal"
        )

        self._prox_cache = SumOfSquaresProxCache(
            atb=atb,
            u=u,
            d=d,
            einsum_path=path,
        )

    @property
    def is_tall(self) -> bool:
        return self.a.shape[0] >= self.a.shape[1]

    def _prox_tall(self, rhs: Array, t: float) -> Array:
        """
        argmin 1/2 |ax-b|^2 + 1/2t |x - v|^2
        a'(ax-b) + 1/t (x -v) = 0
        (a'a + 1/t I) x = 1/t v + a'b
        (a'a + 1/t I) x = 1/t v + a'b
        x = (a'a + 1/t I)^-1 (1/t v + a'b)

        let a = udu' be the eigen decomposition of a'a.
        then:
        (a'a + 1/t I)^-1 = uwu'
        where w = (d + I/t)^-1
        """
        # start = time()
        u = self._prox_cache.u
        d = self._prox_cache.d

        w = 1 / (d + 1 / t)

        x1 = np.einsum(
            "nm,m,km,k->n",
            u,
            w,
            u,
            rhs,
            optimize=self._prox_cache.einsum_path,
        )
        # print(f'{time() - start:.1e}', end='\t\t')
        # start = time()
        # m = self.a.shape[1]
        # q = self.a.T @ self.a + np.eye(m) / t
        # # todo: factorization caching
        # # todo: invert aa' instead if a'a it is faster
        # x2 = np.linalg.solve(q, rhs)
        # print(f'{time() - start:.1e}')
        return x1

    def _prox_fat(self, rhs: Array, t: float) -> Array:
        """
        argmin 1/2 |ax-b|^2 + 1/2t |x - v|^2
        a'(ax-b) + 1/t (x -v) = 0
        (a'a + 1/t I) x = 1/t v + a'b
        (a'a + 1/t I) x = 1/t v + a'b
        x = (a'a + 1/t I)^-1 (1/t v + a'b)

        if a is fat, then we can use the matrix inversion lemma:
        (a'a + 1/t I)^-1 = tI - t^2 a'(I + taa')^-1 a
        = t (I - t a' (I + t a a')^-1 a)

        let a = udu' be the eigen decomposition of aa'.
        then:
        (I + taa')^-1 = uwu' where w = (1 + td)^-1
        so:
        x = t (I - t a'uwu'a) rhs
          = t (rhs - t a'uwu'a rhs)
        """

        # start = time()
        u = self._prox_cache.u
        w = 1 / (1.0 + t * self._prox_cache.d)
        x1 = np.einsum(
            "nm,m,km,k->n", u, w, u, rhs, optimize=self._prox_cache.einsum_path
        )
        x1 = rhs - t * x1
        x1 *= t
        # print(f'{time() - start:.1e}', end='\t\t')
        #
        # start = time()
        # n = self.a.shape[0]
        # q = np.eye(n) + t * self.a @ self.a.T
        # x = np.linalg.solve(q, self.a @ rhs)
        # x *= t
        # x = self.a.T @ x
        # x = rhs - x
        # x *= t
        # print(f'{time() - start:.1e}', end='\t\t')
        #
        # start = time()
        # m = self.a.shape[1]
        # q = self.a.T @ self.a + np.eye(m) / t
        # # todo: factorization caching
        # # todo: invert aa' instead if a'a it is faster
        # x2 = np.linalg.solve(q, rhs)
        # print(f'{time() - start:.1e}')
        return x1

    def prox(self, v: Array, t: float) -> Array:
        if not self._prox_cache:
            self._set_prox_cache()
        atb = self._prox_cache.atb
        rhs = v / t + atb
        if self.is_tall:
            return self._prox_tall(rhs=rhs, t=t)
        else:
            return self._prox_fat(rhs=rhs, t=t)

    def cvxpy_expression(
        self,
        x: cp.Expression,  # type: ignore[name-defined]
    ) -> cp.Expression:  # type: ignore[name-defined]
        return cp.sum_squares(self.a @ x - self.b)  # type: ignore[attr-defined]

    def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
        """
        |ax - b|^2 / 2 = x'a'ax / 2 - b'ax + b'b/2
        :return:
        """
        return ExplicitQuadraticFunction(
            q=MatrixBasedLinearOperator(
                self.a.T @ self.a
            ),  # todo: low rank linear operator
            c=-self.a.T @ self.b,
            d=float((self.b @ self.b) / 2),
        )


class SumOfSquaresLossFactory(LossFactory[SumOfSquaresLoss]):
    def build_loss_function(self, x: Array, y: Array) -> SumOfSquaresLoss:
        return SumOfSquaresLoss(x, y)
