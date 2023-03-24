from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import cvxpy as cp
import numpy as np
import pandas as pd

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
    def build_loss_function(self, x: pd.DataFrame, y: pd.Series) -> L:
        raise NotImplementedError


@dataclass
class SumOfSquaresLoss(QuadraticScalarFunction[Array], ProxableScalarFunction[Array]):
    a: Array
    b: Array

    def __call__(self, x: Array) -> float:
        y = self.a @ x
        r = y - self.b
        return float((r @ r) / 2)

    def prox(self, v: Array, t: float) -> Array:
        """
        argmin 1/2 |ax-b|^2 + 1/2t |x - v|^2
        a'(ax-b) + 1/t (x -v) = 0
        (a'a + 1/t I) x = 1/t v + a'b
        (a'a + 1/t I) x = 1/t v + a'b
        """
        m = self.a.shape[1]
        rhs = v / t + self.a.T @ self.b
        q = self.a.T @ self.a + np.eye(m) / t
        # todo: factorization caching
        # todo: invert aa' instead if a'a it is faster
        return np.linalg.solve(q, rhs)

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
    def build_loss_function(self, x: pd.DataFrame, y: pd.Series) -> SumOfSquaresLoss:
        return SumOfSquaresLoss(x.values, y.values)
