from typing import Protocol

import cvxpy as cp
import pandas as pd
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Minimize, Problem  # type: ignore

from stratified_models.fitters.fitter import (
    Fitter,
    NaiveRefitMixin,
    RefitDataBase,
    Theta,
)
from stratified_models.problem import StratifiedLinearRegressionProblem
from stratified_models.scalar_function import Array, ScalarFunction


class CVXPYScalarFunction(ScalarFunction[Array], Protocol):
    def cvxpy_expression(
        self,
        x: cp.Expression,  # type: ignore[name-defined]
    ) -> cp.Expression:  # type: ignore[name-defined]
        raise NotImplementedError


class CVXPYFitter(Fitter[CVXPYScalarFunction, RefitDataBase], NaiveRefitMixin):
    def fit(
        self, problem: StratifiedLinearRegressionProblem[CVXPYScalarFunction]
    ) -> Theta:
        cvxpy_problem, theta_vars = self._build_cvxpy_problem(problem)
        # todo: verbose flag, select solver
        cvxpy_problem.solve(verbose=True)  # type: ignore
        theta_df = pd.DataFrame(  # todo: should be a common function
            theta_vars.value,
            index=pd.MultiIndex.from_product(
                graph.nodes for graph, _ in problem.graphs
            ),
            columns=problem.regression_features,
        )
        return Theta(df=theta_df), None

    def _build_cvxpy_problem(
        self,
        problem: StratifiedLinearRegressionProblem[CVXPYScalarFunction],
    ) -> tuple[Problem, Variable]:
        theta = Variable(problem.theta_flat_shape())
        loss = self._get_loss(theta, problem)
        local_reg = self._get_local_reg(theta, problem)
        laplace_reg = self._get_laplace_reg(theta, problem)
        cost = loss + local_reg + laplace_reg
        return Problem(Minimize(cost)), theta

    def _get_local_reg(
        self,
        theta: Variable,
        problem: StratifiedLinearRegressionProblem[CVXPYScalarFunction],
    ) -> cp.Expression:  # type: ignore[name-defined]
        return sum(
            (
                func.cvxpy_expression(theta) * gamma
                for func, gamma in problem.regularizers
                if gamma
            )
        )

    def _get_loss(
        self,
        theta: Variable,
        problem: StratifiedLinearRegressionProblem[CVXPYScalarFunction],
    ) -> cp.Expression:  # type: ignore[name-defined]
        loss_expr = 0.0
        for loss, node in problem.loss_iter():
            node_index = problem.get_node_flat_index(node)
            loss_expr += loss.cvxpy_expression(theta[node_index])
        return loss_expr

    def _get_laplace_reg(
        self,
        theta: Variable,
        problem: StratifiedLinearRegressionProblem[CVXPYScalarFunction],
    ) -> cp.Expression:  # type: ignore[name-defined]
        theta_flat = theta.flatten(order="C")
        return sum(
            gamma * laplacian.cvxpy_expression(theta_flat)
            for laplacian, gamma in problem.laplacians()
        )
