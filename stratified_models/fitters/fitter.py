from __future__ import annotations

import itertools
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, TypeVar

import pandas as pd
import scipy

from stratified_models.problem import F, StratifiedLinearRegressionProblem, Theta
from stratified_models.quadratic import ExplicitQuadraticFunction
from stratified_models.scalar_function import Array, QuadraticScalarFunction


@dataclass
class RefitDataBase:
    previous_problem: StratifiedLinearRegressionProblem


RefitDataType = TypeVar("RefitDataType", bound=RefitDataBase)


@dataclass
class ProblemUpdate:
    new_regularization_gammas: list[float]
    new_graph_gammas: list[float]

    # todo: new data? (x and y)

    def apply(
        self, problem: StratifiedLinearRegressionProblem
    ) -> StratifiedLinearRegressionProblem:
        return StratifiedLinearRegressionProblem(
            x=problem.x,
            y=problem.y,
            loss_factory=problem.loss_factory,
            regularizers=[
                (f, gamma)
                for (f, _), gamma in zip(
                    problem.regularizers, self.new_regularization_gammas
                )
            ],
            graphs=[
                (graph, gamma)
                for (graph, _), gamma in zip(problem.graphs, self.new_graph_gammas)
            ],
            regression_features=problem.regression_features,
        )


class Fitter(Protocol[F, RefitDataType]):
    def fit(
        self,
        problem: StratifiedLinearRegressionProblem[F],
    ) -> tuple[Theta, RefitDataType, float]:
        raise NotImplementedError

    def refit(
        self,
        problem_update: ProblemUpdate,
        refit_data: RefitDataType,
    ) -> tuple[Theta, RefitDataType, float]:
        raise NotImplementedError


Q = TypeVar("Q", bound=QuadraticScalarFunction[Array])


# todo: not really a mixin, but a baseclass.
#  I'd be happier if it were a mixin but not sure how
#  to specify it has a fit method
class NaiveRefitMixin:
    def refit(
        self,
        problem_update: ProblemUpdate,
        refit_data: RefitDataBase,
    ) -> tuple[Theta, RefitDataBase]:
        new_problem = problem_update.apply(refit_data.previous_problem)
        return self.fit(new_problem)


# @dataclass
# class QuadraticRefitData:
#     previous_problem: StratifiedLinearRegressionProblem
#     quadratic: ExplicitQuadraticFunction
#     previous_solutions: list[Array]
#
@dataclass
class QuadraticRefitData(RefitDataBase):
    previous_quadratic_components: list[ExplicitQuadraticFunction, float]


# todo: wrong location
# todo: impl refit, return cost and refit data
@dataclass
class QuadraticProblemFitter(Fitter[Q, RefitDataBase], NaiveRefitMixin):
    solver: PSDSystemSolver

    def fit(
        self,
        problem: StratifiedLinearRegressionProblem[Q],
    ) -> tuple[Theta, RefitDataType, float,]:
        f, components = self._build_quadratic(problem)
        theta = self.solver.solve(f)
        cost = f(theta)
        theta = theta.reshape((-1, problem.m))
        theta_df = pd.DataFrame(  # todo: should be a common function
            theta,
            index=pd.MultiIndex.from_product(
                graph.nodes for graph, _ in problem.graphs
            ),
            columns=problem.regression_features,
        )
        refit_data = QuadraticRefitData(
            previous_problem=problem,
            previous_quadratic_components=components,
        )
        return Theta(df=theta_df, shape=problem.theta_shape()), refit_data, cost

    def refit(
        self,
        problem_update: ProblemUpdate,
        refit_data: QuadraticRefitData,
    ) -> tuple[Theta, QuadraticRefitData, float]:
        problem = problem_update.apply(refit_data.previous_problem)
        f, components = self._build_quadratic_from_previous(
            problem_update=problem_update,
            refit_data=refit_data,
        )
        # todo: pass previous solutions (or even factorizations to use preconditioners)
        #  to PSDSolver. implement a resolve in PSDSolver.
        theta = self.solver.solve(f)
        cost = f(theta)
        theta = theta.reshape((-1, problem.m))
        theta_df = pd.DataFrame(  # todo: should be a common function
            theta,
            index=pd.MultiIndex.from_product(
                graph.nodes for graph, _ in problem.graphs
            ),
            columns=problem.regression_features,
        )
        refit_data = QuadraticRefitData(
            previous_problem=problem,
            previous_quadratic_components=refit_data,
        )
        return Theta(df=theta_df, shape=problem.theta_shape()), refit_data, cost

    def _build_quadratic(
        self,
        problem: StratifiedLinearRegressionProblem[Q],
    ) -> tuple[ExplicitQuadraticFunction, list[ExplicitQuadraticFunction, float]]:
        k, m = problem.theta_flat_shape()
        cost_components = []

        # loss
        loss_components = {}
        for node, x, y in problem.node_data_iter():
            i = problem.get_node_flat_index(node)
            loss = problem.loss_factory.build_loss_function(
                x[problem.regression_features].values, y.values
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

        return (
            ExplicitQuadraticFunction.sum(
                m=k * m,
                components=cost_components,
            ),
            cost_components,
        )

    def _build_quadratic_from_previous(
        self,
        problem_update: ProblemUpdate,
        refit_data: QuadraticRefitData,
    ) -> tuple[ExplicitQuadraticFunction, list[ExplicitQuadraticFunction, float],]:
        components = [
            (cost_component, new_gamma)
            for new_gamma, (cost_component, _) in zip(
                itertools.chain(
                    [1.0],
                    problem_update.new_regularization_gammas,
                    problem_update.new_graph_gammas,
                ),
                refit_data.previous_quadratic_components,
            )
        ]
        k, m = refit_data.previous_problem.theta_flat_shape()
        return ExplicitQuadraticFunction.sum(k * m, components=components), components


# todo: wrong location
class PSDSystemSolver:
    @abstractmethod
    def solve(
        self,
        f: ExplicitQuadraticFunction,
    ) -> Array:
        pass


# todo: wrong location
class DirectSolver(PSDSystemSolver):
    def solve(
        self,
        f: ExplicitQuadraticFunction,
    ) -> Array:
        a = f.q.as_sparse_matrix()
        return scipy.sparse.linalg.spsolve(a, -f.c)  # type: ignore[no-any-return]


@dataclass
class CGSolver(PSDSystemSolver):
    tol: float = 1e-6
    max_iter: Optional[int] = None

    def solve(
        self,
        f: ExplicitQuadraticFunction,
    ) -> Array:
        theta, info = scipy.sparse.linalg.cg(
            f.q.to_scipy_linear_operator(),
            -f.c,
            tol=self.tol,
            maxiter=self.max_iter,
        )
        # todo: raise error if info indicates a problem
        return theta  # type: ignore[no-any-return]


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
