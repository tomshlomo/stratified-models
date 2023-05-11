from __future__ import annotations

import itertools
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import scipy

from stratified_models.fitters.fitter import Fitter, ProblemUpdate, RefitDataBase
from stratified_models.problem import StratifiedLinearRegressionProblem, Theta
from stratified_models.quadratic import ExplicitQuadraticFunction
from stratified_models.scalar_function import Array, QuadraticScalarFunction


@dataclass
class QuadraticRefitData(RefitDataBase[QuadraticScalarFunction[Array]]):
    previous_quadratic_components: list[tuple[ExplicitQuadraticFunction, float]]


@dataclass
class QuadraticProblemFitter(
    Fitter[QuadraticScalarFunction[Array], QuadraticRefitData]
):
    solver: PSDSystemSolver

    def fit(
        self,
        problem: StratifiedLinearRegressionProblem[QuadraticScalarFunction[Array]],
    ) -> tuple[Theta, QuadraticRefitData, float]:
        f, components = self._build_quadratic(problem)
        theta = self.solver.solve(f)
        cost = f(theta)
        theta = theta.reshape((-1, problem.m))
        refit_data = QuadraticRefitData(
            previous_problem=problem,
            previous_quadratic_components=components,
        )
        return (
            Theta.from_array(arr=theta, problem=problem),
            refit_data,
            cost,
        )

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
        # todo: pass previous solutions (and even factorizations to use as
        #  preconditioners) to PSDSolver. implement a resolve in PSDSolver.
        theta = self.solver.solve(f)
        cost = f(theta)
        theta = theta.reshape((-1, problem.m))
        refit_data = QuadraticRefitData(
            previous_problem=problem,
            previous_quadratic_components=components,
        )
        return (
            Theta.from_array(arr=theta, problem=refit_data.previous_problem),
            refit_data,
            cost,
        )

    def _build_quadratic(
        self,
        problem: StratifiedLinearRegressionProblem[QuadraticScalarFunction[Array]],
    ) -> tuple[
        ExplicitQuadraticFunction, list[tuple[ExplicitQuadraticFunction, float]]
    ]:
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
    ) -> tuple[
        ExplicitQuadraticFunction,
        list[tuple[ExplicitQuadraticFunction, float]],
    ]:
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


class PSDSystemSolver:
    @abstractmethod
    def solve(
        self,
        f: ExplicitQuadraticFunction,
    ) -> Array:
        pass


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
