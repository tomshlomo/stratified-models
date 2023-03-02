from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from stratified_models.fitters.protocols import Theta
from stratified_models.problem import F, StratifiedLinearRegressionProblem
from stratified_models.quadratic import ExplicitQuadraticFunction
from stratified_models.scalar_function import QuadraticScalarFunction


class Fitter(Generic[F]):
    @abstractmethod
    def fit(self, problem: StratifiedLinearRegressionProblem[F]) -> Theta:
        pass


Q = TypeVar("Q", bound=QuadraticScalarFunction)


# todo: wrong location
@dataclass
class QuadraticProblemFitter:
    solver: PSDSystemSolver

    def fit(self, problem: StratifiedLinearRegressionProblem[Q]):
        f = self._build_quadratic(problem)
        theta = self.solver.solve(f, problem)
        theta = theta.reshape((-1, problem.m))
        theta_df = pd.DataFrame(  # todo: should be a common function
            theta,
            index=pd.MultiIndex.from_product(
                graph.nodes for graph, _ in problem.graphs
            ),
            columns=problem.regression_features,
        )
        return Theta(df=theta_df)

    def _build_quadratic(
        self, problem: StratifiedLinearRegressionProblem[Q]
    ) -> ExplicitQuadraticFunction:
        k, m = problem.theta_flat_shape()
        cost_components = []

        # loss
        loss_components = {}
        for node, x, y in problem.node_data_iter():
            i = problem.get_node_flat_index(node)
            loss = problem.loss_factory.build_loss_function(
                x, y
            ).to_explicit_quadratic()
            loss_components[i] = loss
        cost_components.append(
            (
                ExplicitQuadraticFunction.concat(k, m, loss_components),
                1.0,
            )
        )

        # local reg
        cost_components.extend(
            (reg.to_explicit_quadratic().repeat(k), gamma)
            for reg, gamma in problem.regularizers
        )

        # laplacian reg
        cost_components.extend(
            (laplacian.to_explicit_quadratic(), gamma)
            for laplacian, gamma in problem.laplacians()
        )

        return ExplicitQuadraticFunction.sum(
            m=k * m,
            components=cost_components,
        )


# todo: wrong location
class PSDSystemSolver:
    @abstractmethod
    def solve(
        self,
        f: ExplicitQuadraticFunction,
        problem: StratifiedLinearRegressionProblem[Q],
    ) -> npt.NDArray[np.float64]:
        pass


# todo: wrong location
class DirectSolver(PSDSystemSolver):
    def solve(
        self,
        f: ExplicitQuadraticFunction,
        problem: StratifiedLinearRegressionProblem[Q],
    ) -> npt.NDArray[np.float64]:
        a = f.q.as_sparse_matrix()
        return scipy.sparse.linalg.spsolve(a, -f.c)


@dataclass
class CGSolver(PSDSystemSolver):
    tol: float = 1e-6
    max_iter: Optional[int] = None

    def solve(
        self,
        f: ExplicitQuadraticFunction,
        problem: StratifiedLinearRegressionProblem[Q],
    ) -> npt.NDArray[np.float64]:
        theta, info = scipy.sparse.linalg.cg(
            f.q.to_scipy_linear_operator(),
            -f.c,
            tol=self.tol,
            maxiter=self.max_iter,
        )
        # todo: raise error if info indicates a problem
        return theta


# @dataclass
# class Costs:
#     loss: float
#     local_regularization: float
#     laplacian_regularization: float
#
#     @classmethod
#     def from_problem_and_theta(
#         cls,
#         problem: QuadraticStratifiedLinearRegressionProblem[Node],
#         theta: Theta[Node],
#     ) -> Costs:
#         loss = 0.0
#         theta_df = theta.df.set_index(pd.MultiIndex.from_tuples(theta.df.index))
#         for node, node_data in problem.nodes_data.items():
#             y_pred = node_data.x @ theta_df.loc[node, :].values
#             d = y_pred - node_data.y
#             loss += d @ d
#         local_reg = theta.df.values.ravel() @ theta.df.values.ravel() * problem.l2_reg
#         laplace_reg = problem.graph.laplacian_quad_form(theta_df.values)
#         return Costs(
#             loss=loss,
#             local_regularization=local_reg,
#             laplacian_regularization=laplace_reg,
#         )
#
#     def total(self) -> float:
#         return self.loss + self.local_regularization + self.laplacian_regularization
