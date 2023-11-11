from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
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
    X,
)

L = TypeVar("L", bound=ScalarFunction[Array], covariant=True)

DenseOrSparseMatrix = Union[Array, scipy.sparse.spmatrix]


class LossFactory(Generic[L]):
    @abstractmethod
    def build_loss_function(self, x: DenseOrSparseMatrix, y: Array) -> L:
        raise NotImplementedError


EinsumPath = list[Union[str, tuple[int, ...]]]


@dataclass
class SumOfSquaresProxCache:
    atb: Array
    u: Array
    d: Array
    einsum_path: EinsumPath


def _to_numpy_array(x: Union[Array, scipy.sparse.spmatrix]) -> Array:
    return (  # type:ignore[no-any-return]
        x.toarray() if isinstance(x, scipy.sparse.spmatrix) else x
    )


@dataclass
class SumOfSquaresLoss(ProxableScalarFunction[Array], QuadraticScalarFunction[Array]):
    a: DenseOrSparseMatrix
    b: Array
    _prox_cache: Optional[SumOfSquaresProxCache] = None

    def __call__(self, x: Array) -> float:
        y = self.a @ x
        r = y - self.b
        return float((r @ r) / 2)

    def _get_prox_cache(self) -> SumOfSquaresProxCache:
        if self._prox_cache is not None:
            return self._prox_cache
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
        return self._prox_cache

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
        prox_cache = self._get_prox_cache()
        u = prox_cache.u
        d = prox_cache.d

        w = 1 / (d + 1 / t)

        x1 = np.einsum(
            "nm,m,km,k->n",
            u,
            w,
            u,
            rhs,
            optimize=prox_cache.einsum_path,
        )
        return x1  # type:ignore[no-any-return]

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

        prox_cache = self._get_prox_cache()
        u = prox_cache.u
        w = 1 / (1.0 + t * prox_cache.d)
        x1 = np.einsum("nm,m,km,k->n", u, w, u, rhs, optimize=prox_cache.einsum_path)
        x1 = rhs - t * x1
        x1 *= t
        return x1  # type:ignore[no-any-return]

    def prox(self, v: Array, t: float) -> Array:
        if t == 0:
            return v
        prox_cache = self._get_prox_cache()
        atb = prox_cache.atb
        rhs = v / t + atb
        if self.is_tall:
            return self._prox_tall(rhs=rhs, t=t)
        else:
            return self._prox_fat(rhs=rhs, t=t)

    def cvxpy_expression(
        self,
        x: cp.Expression,  # type: ignore[name-defined]
    ) -> cp.Expression:  # type: ignore[name-defined]
        return cp.sum_squares(self.a @ x - self.b) / 2  # type: ignore[attr-defined]

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
    def build_loss_function(self, x: DenseOrSparseMatrix, y: Array) -> SumOfSquaresLoss:
        return SumOfSquaresLoss(x, y)


@dataclass
class LogisticOverLinear(ProxableScalarFunction[Array]):
    """
    f(x) = sum_i log(1 + exp(a_i'x))
    """

    # todo: can be generalized to general binary classification l(x;a,y) = p(yax)
    a: Union[Array, scipy.sparse.spmatrix]
    _prox_cache: Optional[SumOfSquaresProxCache] = None

    def __call__(self, x: Array) -> float:
        losses = np.log1p(np.exp(self.a @ x))
        return float(losses.sum())

    def prox(self, v: X, t: float) -> X:
        # todo: more control over optimizer (method, memory, tolerances, and so on)
        result = scipy.optimize.minimize(
            partial(self._prox_eval_with_grad),
            x0=v,
            args=(v, 1 / t),
            method="L-BFGS-B",
            jac=True,
        )
        return result.x  # type:ignore[no-any-return]

    def _prox_eval_with_grad(
        self, x: Array, v: Array, rho: float
    ) -> tuple[float, Array]:
        """
        h(x) = sum_i log(1 + exp(a_i'x) + rho/2 norm(x - v)^2
        grad h(x) = sum_i a_i exp(a_i'x) / (1 + exp(a_i'x) + rho (x - v)
        """
        exp_z = np.exp(self.a @ x)
        h = np.log1p(exp_z).sum()
        sig_z = exp_z / (1 + exp_z)
        grad_h = sig_z @ self.a

        x2v = x - v
        grad_h += rho * x2v
        h += (x2v @ x2v) * rho / 2
        return h, grad_h

    def cvxpy_expression(
        self,
        x: cp.Expression,  # type: ignore[name-defined]
    ) -> cp.Expression:  # type: ignore[name-defined]
        losses = cp.logistic(self.a @ x)  # type: ignore[attr-defined]
        return cp.sum(losses)  # type: ignore[attr-defined]


class LogisticLossFactory(LossFactory[LogisticOverLinear]):
    def build_loss_function(
        self, x: DenseOrSparseMatrix, y: Array
    ) -> LogisticOverLinear:
        y = -y[:, np.newaxis]
        a = x.multiply(y) if isinstance(x, scipy.sparse.spmatrix) else x * y
        return LogisticOverLinear(a=a)
