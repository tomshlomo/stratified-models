from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import cvxpy as cp
import numpy as np
import pandas as pd

from stratified_models.linear_operator import MatrixBasedLinearOperator
from stratified_models.quadratic import ExplicitQuadraticFunction
from stratified_models.scalar_function import DenseMatrix, ScalarFunction, Vector

L = TypeVar("L", bound=ScalarFunction)


class LossFactory(Generic[L]):
    @abstractmethod
    def build_loss_function(self, x: pd.DataFrame, y: pd.Series) -> L:
        pass


class SumOfSquaresLossFactory(LossFactory):
    @abstractmethod
    def build_loss_function(self, x: pd.DataFrame, y: pd.Series) -> SumOfSquaresLoss:
        return SumOfSquaresLoss(x.values, y.values)


@dataclass
class SumOfSquaresLoss:
    a: DenseMatrix
    b: Vector

    def __call__(self, x: Vector) -> float:
        y = self.a @ x
        r = y - self.b
        return (r @ r) / 2

    def prox(self, v: Vector, t: float) -> Vector:
        """
        argmin 1/2 |ax-b|^2 + rho/2 |x - v|^2
        a'(ax-b) + rho (x -v) = 0
        (a'a + rho I) x = rho v + a'b
        (a'a/rho + I) x = v + a'b/rho
        """
        m = self.a.shape[1]
        rhs = v + self.a.T @ self.b * t
        q = self.a.T @ self.a + np.eye(m)
        # todo: factorization caching
        # todo: invert aa' instead if a'a it is faster
        return np.linalg.solve(q, rhs)

    def cvxpy_expression(self, x: cp.Expression) -> cp.Expression:
        return cp.sum_squares(self.a @ x - self.b)

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
            d=(self.b @ self.b) / 2,
        )
