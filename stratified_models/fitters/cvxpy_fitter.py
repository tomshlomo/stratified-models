import itertools
from dataclasses import dataclass
from typing import Protocol

from cvxpy.expressions.constants import Parameter  # type:ignore[attr-defined]
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Minimize, Problem  # type:ignore[attr-defined]

from stratified_models.fitters.fitter import Fitter, ProblemUpdate, RefitDataBase
from stratified_models.problem import StratifiedLinearRegressionProblem, Theta
from stratified_models.scalar_function import Array, ScalarFunction


class CVXPYScalarFunction(ScalarFunction[Array], Protocol):
    def cvxpy_expression(
        self,
        x: Expression,
    ) -> Expression:
        raise NotImplementedError


@dataclass
class CVXPYRefitData(RefitDataBase[CVXPYScalarFunction]):
    previous_cvxpy_problem: Problem
    local_reg_params: list[Parameter]
    laplace_params: list[Parameter]
    theta_vars: Variable


class CVXPYFitter(Fitter[CVXPYScalarFunction, CVXPYRefitData]):
    def fit(
        self, problem: StratifiedLinearRegressionProblem[CVXPYScalarFunction]
    ) -> tuple[Theta, CVXPYRefitData, float]:
        cvxpy_problem, theta_vars, refit_data = self._build_cvxpy_problem(problem)
        # todo: verbose flag, select solver
        cvxpy_problem.solve(verbose=True)  # type:ignore[no-untyped-call]
        return (
            Theta.from_array(arr=theta_vars.value, problem=problem),
            refit_data,
            cvxpy_problem.value,
        )

    def refit(
        self, problem_update: ProblemUpdate, refit_data: CVXPYRefitData
    ) -> tuple[Theta, CVXPYRefitData, float]:
        for param, new_val in itertools.chain(
            zip(
                refit_data.local_reg_params,
                problem_update.new_regularization_gammas,
            ),
            zip(
                refit_data.laplace_params,
                problem_update.new_graph_gammas,
            ),
        ):
            param.value = new_val
        refit_data.previous_cvxpy_problem.solve(
            verbose=True
        )  # type:ignore[no-untyped-call]
        return (
            Theta.from_array(
                arr=refit_data.theta_vars.value, problem=refit_data.previous_problem
            ),
            refit_data,
            refit_data.previous_cvxpy_problem.value,
        )

    def _build_cvxpy_problem(
        self,
        problem: StratifiedLinearRegressionProblem[CVXPYScalarFunction],
    ) -> tuple[Problem, Variable, CVXPYRefitData]:
        theta = Variable(problem.theta_flat_shape())
        loss = self._get_loss(theta, problem)
        local_reg, local_reg_params = self._get_local_reg(theta, problem)
        laplace_reg, laplace_params = self._get_laplace_reg(theta, problem)
        cost = loss + local_reg + laplace_reg
        cvxpy_problem = Problem(Minimize(cost))
        return (
            cvxpy_problem,
            theta,
            CVXPYRefitData(
                previous_problem=problem,
                previous_cvxpy_problem=cvxpy_problem,
                local_reg_params=local_reg_params,
                laplace_params=laplace_params,
                theta_vars=theta,
            ),
        )

    def _get_local_reg(
        self,
        theta: Variable,
        problem: StratifiedLinearRegressionProblem[CVXPYScalarFunction],
    ) -> tuple[Expression, list[Parameter]]:
        expr = 0.0
        params = []
        for func, gamma in problem.regularizers():
            param = Parameter(value=gamma, nonneg=True)
            expr += param * func.cvxpy_expression(theta)
            params.append(param)
        return expr, params  # type:ignore[return-value]

    def _get_loss(
        self,
        theta: Variable,
        problem: StratifiedLinearRegressionProblem[CVXPYScalarFunction],
    ) -> Expression:
        loss_expr = 0.0
        for loss, node in problem.loss_iter():
            node_index = problem.get_node_flat_index(node)
            loss_expr += loss.cvxpy_expression(theta[node_index])
        return loss_expr  # type:ignore[return-value]

    def _get_laplace_reg(
        self,
        theta: Variable,
        problem: StratifiedLinearRegressionProblem[CVXPYScalarFunction],
    ) -> tuple[Expression, list[Parameter]]:
        expr = 0.0
        params = []
        for laplacian, gamma in problem.laplacians():
            param = Parameter(value=gamma, nonneg=True)
            expr += param * laplacian.cvxpy_expression(theta)
            params.append(param)
        return expr, params  # type:ignore[return-value]
