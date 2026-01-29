from __future__ import annotations

from collections.abc import Mapping

import attrs
import cvxpy as cp
import jax.numpy as jnp
import structlog

from stratified_models.model import StartifiedModel, ThetaShape
from stratified_models.problem import (
    AbstractProblem,
    Hyperparameters,
    SolveInfo,
    Solver,
)
from stratified_models.scalar_function import CVXPYScalarFunction

logger = structlog.get_logger()


@attrs.frozen(kw_only=True)
class CVXPYSolveInfo(SolveInfo):
    status: str
    value: float | None
    solver_stats: object | None
    extra: Mapping[str, object] = attrs.field(factory=dict)

    def converged(self) -> bool:
        return self.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


@attrs.frozen(kw_only=True)
class CompiledCVXPYProblem:
    cvxpy_problem: cp.Problem
    theta: cp.Variable
    local_reg_params: Mapping[str, cp.Parameter]
    laplace_params: Mapping[str, cp.Parameter]
    loss: cp.Expression
    local_reg: cp.Expression
    laplace_reg: cp.Expression
    theta_shape: ThetaShape


@attrs.frozen(kw_only=True)
class CVXPYSolver(Solver[CompiledCVXPYProblem, CVXPYSolveInfo]):
    verbose: bool = False

    def compile(self, abstract_problem: AbstractProblem) -> CompiledCVXPYProblem:
        logger.debug(
            "cvxpy_compile_start",
            theta_shape=abstract_problem.theta_shape.array_shape,
            num_graphs=len(abstract_problem.graphs),
            num_regularizers=len(abstract_problem.regularizers),
        )
        theta = cp.Variable(abstract_problem.theta_shape.array_shape)
        loss = self._get_loss(theta, abstract_problem)
        local_reg, local_reg_params = self._get_local_reg(theta, abstract_problem)
        laplace_reg, laplace_params = self._get_laplace_reg(theta, abstract_problem)
        cost = loss + local_reg + laplace_reg
        cvxpy_problem = cp.Problem(cp.Minimize(cost))
        logger.debug(
            "cvxpy_compile_complete",
            num_local_reg_params=len(local_reg_params),
            num_laplace_params=len(laplace_params),
        )
        return CompiledCVXPYProblem(
            cvxpy_problem=cvxpy_problem,
            theta=theta,
            local_reg_params=local_reg_params,
            laplace_params=laplace_params,
            loss=loss,
            local_reg=local_reg,
            laplace_reg=laplace_reg,
            theta_shape=abstract_problem.theta_shape,
        )

    def solve(
        self,
        compiled_problem: CompiledCVXPYProblem,
        hyperparameters: Hyperparameters,
    ) -> tuple[StartifiedModel, CVXPYSolveInfo]:
        logger.debug(
            "cvxpy_solve_start",
            num_regularizers=len(compiled_problem.local_reg_params),
            num_graphs=len(compiled_problem.laplace_params),
            verbose=self.verbose,
        )
        for name, param in compiled_problem.local_reg_params.items():
            param.value = hyperparameters.regularizers[name]
        for name, param in compiled_problem.laplace_params.items():
            param.value = hyperparameters.graphs[name]

        compiled_problem.cvxpy_problem.solve(verbose=self.verbose)

        status = str(compiled_problem.cvxpy_problem.status)
        converged = status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        value = (
            None
            if compiled_problem.cvxpy_problem.value is None
            else float(compiled_problem.cvxpy_problem.value)
        )
        logger.info(
            "cvxpy_solve_status",
            status=status,
            converged=converged,
            has_value=value is not None,
        )

        info = CVXPYSolveInfo(
            status=status,
            value=value,
            solver_stats=compiled_problem.cvxpy_problem.solver_stats,
            extra={"converged": converged},
        )
        model = StartifiedModel(
            theta=jnp.asarray(compiled_problem.theta.value),
            shape=compiled_problem.theta_shape,
        )
        return model, info

    def _get_local_reg(
        self,
        theta: cp.Variable,
        problem: AbstractProblem,
    ) -> tuple[cp.Expression, Mapping[str, cp.Parameter]]:
        expr = 0.0
        params = {}
        for name, func in problem.regularizers.items():
            params[name] = cp.Parameter(nonneg=True, name=name)
            assert isinstance(func, CVXPYScalarFunction)
            expr += params[name] * func.cvxpy_expression(theta)
        return expr, params  # ty:ignore[invalid-return-type]

    def _get_loss(
        self,
        theta: cp.Variable,
        problem: AbstractProblem,
    ) -> cp.Expression:
        loss_expr = 0.0
        for node, loss in problem.group_losses():
            node_index = problem.theta_shape.node_to_index(node)
            assert isinstance(loss, CVXPYScalarFunction)
            loss_expr += loss.cvxpy_expression(theta[node_index])
        return loss_expr  # type:ignore[return-value]

    def _get_laplace_reg(
        self,
        theta: cp.Variable,
        problem: AbstractProblem,
    ) -> tuple[cp.Expression, Mapping[str, cp.Parameter]]:
        expr = 0.0
        params = {}
        for name, laplacian in problem.laplacians():
            params[name] = cp.Parameter(nonneg=True, name=name)
            assert isinstance(laplacian, CVXPYScalarFunction)
            expr += params[name] * laplacian.cvxpy_expression(theta)
        return expr, params  # type:ignore[return-value]
