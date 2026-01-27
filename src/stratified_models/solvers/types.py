from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property

import attrs
import jax
import jax.numpy as jnp
import pandas as pd
import structlog

from stratified_models.graph import RegularizationGraph
from stratified_models.loss import Loss
from stratified_models.model import Node, StartifiedModel, ThetaShape
from stratified_models.scalar_function import ScalarFunction

logger = structlog.get_logger()


class SolveInfo(ABC):
    @abstractmethod
    def converged(self) -> bool:
        pass


@attrs.frozen(kw_only=True)
class Hyperparameters:
    graphs: Mapping[str, float]
    regularizers: Mapping[str, float]


@attrs.frozen(kw_only=True)
class Objectives:
    loss: jax.Array
    regularizers: Mapping[str, jax.Array]
    laplacians: Mapping[str, jax.Array]

    def total(self, hyperparameters: Hyperparameters) -> jax.Array:
        regularizers_total = jnp.asarray(0.0)
        for name, value in self.regularizers.items():
            regularizers_total = (
                regularizers_total + hyperparameters.regularizers[name] * value
            )

        laplacians_total = jnp.asarray(0.0)
        for name, value in self.laplacians.items():
            laplacians_total = laplacians_total + hyperparameters.graphs[name] * value

        return self.loss + regularizers_total + laplacians_total


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
        logger.debug(
            "abstract_problem_group_data_start",
            groupby_cols=[str(col) for col in groupby_cols],
            rows=self.n,
        )
        for node, x_slice in self.x.groupby(groupby_cols):
            logger.debug(
                "abstract_problem_group_data_group",
                node=tuple(str(part) for part in node),
                rows=x_slice.shape[0],
            )
            yield node, x_slice, self.y[x_slice.index]

    def group_losses(self) -> Iterable[tuple[Node, ScalarFunction]]:
        logger.debug("abstract_problem_group_losses_start", rows=self.n)
        for node, x, y in self.group_data():
            logger.debug(
                "abstract_problem_group_losses_group",
                node=tuple(str(part) for part in node),
                rows=x.shape[0],
            )
            yield (
                node,
                self.loss.build(
                    jnp.asarray(x[self.regression_features].to_numpy()),
                    jnp.asarray(y.to_numpy()),
                ),
            )

    def laplacians(self) -> Iterable[tuple[str, ScalarFunction]]:
        logger.debug(
            "abstract_problem_laplacians_start",
            num_graphs=len(self.graphs),
            dims=self.theta_shape.array_shape,
        )
        for i, graph in enumerate(self.graphs):
            logger.debug(
                "abstract_problem_laplacians_graph",
                graph_index=i,
                graph_name=str(graph.name),
                graph_size=graph.size,
            )
            yield (
                str(graph.name),
                graph.laplacian(
                    axis=i,
                    dims=self.theta_shape.array_shape,
                ),
            )

    def objectives(self, model: StartifiedModel) -> Objectives:
        logger.debug(
            "abstract_problem_objectives_start",
            num_regularizers=len(self.regularizers),
            num_graphs=len(self.graphs),
        )
        loss_value = jnp.asarray(0.0)
        for node, loss in self.group_losses():
            theta_node = model.theta_at_node(node)
            loss_value = loss_value + loss(theta_node)

        regularizers = {
            name: reg(model.theta) for name, reg in self.regularizers.items()
        }

        laplacians: dict[str, jax.Array] = {}
        for name, laplacian in self.laplacians():
            laplacians[name] = laplacian(model.theta)

        logger.debug(
            "abstract_problem_objectives_complete",
            num_regularizers=len(regularizers),
            num_laplacians=len(laplacians),
        )
        return Objectives(
            loss=loss_value,
            regularizers=regularizers,
            laplacians=laplacians,
        )


class Solver[T, I: SolveInfo](ABC):
    @abstractmethod
    def compile(self, abstract_problem: AbstractProblem) -> T:
        pass

    @abstractmethod
    def solve(
        self,
        compiled_problem: T,
        hyperparameters: Hyperparameters,
    ) -> tuple[StartifiedModel, I]:
        pass

    def compile_and_solve(
        self,
        abstract_problem: AbstractProblem,
        hyperparameters: Hyperparameters,
    ) -> tuple[StartifiedModel, I, T]:
        logger.info(
            "solver_compile_start",
            solver=type(self).__name__,
            rows=abstract_problem.n,
            num_graphs=len(abstract_problem.graphs),
            num_regularizers=len(abstract_problem.regularizers),
        )
        compiled_problem = self.compile(abstract_problem)
        logger.info("solver_compile_complete", solver=type(self).__name__)
        logger.info("solver_solve_start", solver=type(self).__name__)
        model, info = self.solve(compiled_problem, hyperparameters)
        logger.info(
            "solver_solve_complete",
            solver=type(self).__name__,
            converged=info.converged(),
            solve_info=type(info).__name__,
        )
        return model, info, compiled_problem
