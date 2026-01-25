from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property

import attrs
import cvxpy as cp
import pandas as pd

from stratified_models.simpler.graph import RegularizationGraph
from stratified_models.simpler.loss import Loss
from stratified_models.simpler.model import Node, StartifiedModel, ThetaShape
from stratified_models.simpler.scalar_function import (
    CVXPYScalarFunction,
    ScalarFunction,
)


@attrs.frozen(kw_only=True)
class Hyperparameters:
    graphs: Mapping[str, float]
    regularizers: Mapping[str, float]


@attrs.frozen(kw_only=True)
class Objectives:
    loss: float
    regularizers: Mapping[str, float]
    laplacians: Mapping[str, float]

    def total(self, hyperparameters: Hyperparameters) -> float:
        regularizers = sum(
            hyperparameters.regularizers[name] * value
            for name, value in self.regularizers.items()
        )
        laplacians = sum(
            hyperparameters.graphs[name] * value
            for name, value in self.laplacians.items()
        )
        return float(self.loss + regularizers + laplacians)


@attrs.frozen(kw_only=True)
class AbstractProblem:
    x: pd.DataFrame
    y: pd.Series
    regression_features: Sequence[str]
    graphs: Sequence[RegularizationGraph]
    loss: Loss
    regularizers: Mapping[str, ScalarFunction]

    @property
    def n(self) -> int:
        return self.x.shape[0]

    @cached_property
    def theta_shape(self) -> ThetaShape:
        return ThetaShape(
            regression_features=self.regression_features,
            stratifications=[graph.stratification for graph in self.graphs],
        )

    def group_data(self) -> Iterable[tuple[Node, pd.DataFrame, pd.Series]]:
        groupby_cols = list(self.theta_shape.stratification_features)
        for node, x_slice in self.x.groupby(groupby_cols):
            yield node, x_slice, self.y[x_slice.index]

    def group_losses(self) -> Iterable[tuple[Node, Loss]]:
        for node, x, y in self.group_data():
            yield (
                node,
                self.loss.build(x[self.regression_features].to_numpy(), y.to_numpy()),
            )

    def laplacians(self) -> Iterable[tuple[str, ScalarFunction]]:
        for i, graph in enumerate(self.graphs):
            yield (
                str(graph.name),
                graph.laplacian(
                    axis=i,
                    dims=self.theta_shape.array_shape,
                ),
            )

    def objectives(self, model: StartifiedModel) -> Objectives:
        loss_value = 0.0
        for node, loss in self.group_losses():
            theta_node = model.theta_at_node(node)
            loss_value += loss(theta_node)  # type:ignore[operator]

        regularizers = {
            name: float(reg(model.theta)) for name, reg in self.regularizers.items()
        }

        laplacians: dict[str, float] = {}
        for name, laplacian in self.laplacians():
            laplacians[name] = float(laplacian(model.theta))

        return Objectives(
            loss=float(loss_value),
            regularizers=regularizers,
            laplacians=laplacians,
        )


class Solver[T](ABC):
    @abstractmethod
    def compile(self, abstract_problem: AbstractProblem) -> T:
        pass

    @abstractmethod
    def solve(
        self, compiled_problem: T, hyperparameters: Hyperparameters
    ) -> StartifiedModel:
        pass

    def compile_and_solve(
        self, abstract_problem: AbstractProblem, hyperparameters: Hyperparameters
    ) -> tuple[StartifiedModel, T]:
        compiled_problem = self.compile(abstract_problem)
        return self.solve(compiled_problem, hyperparameters), compiled_problem


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
class CVXPYSolver(Solver[CompiledCVXPYProblem]):
    verbose: bool = False

    def compile(self, abstract_problem: AbstractProblem) -> CompiledCVXPYProblem:
        theta = cp.Variable(abstract_problem.theta_shape.array_shape)
        loss = self._get_loss(theta, abstract_problem)
        local_reg, local_reg_params = self._get_local_reg(theta, abstract_problem)
        laplace_reg, laplace_params = self._get_laplace_reg(theta, abstract_problem)
        cost = loss + local_reg + laplace_reg
        cvxpy_problem = cp.Problem(cp.Minimize(cost))
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
        self, compiled_problem: CompiledCVXPYProblem, hyperparameters: Hyperparameters
    ) -> StartifiedModel:
        for name, param in compiled_problem.local_reg_params.items():
            param.value = hyperparameters.regularizers[name]
        for name, param in compiled_problem.laplace_params.items():
            param.value = hyperparameters.graphs[name]

        compiled_problem.cvxpy_problem.solve(verbose=self.verbose)
        return StartifiedModel(
            theta=compiled_problem.theta.value,  # ty:ignore[invalid-argument-type]
            shape=compiled_problem.theta_shape,
        )

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
