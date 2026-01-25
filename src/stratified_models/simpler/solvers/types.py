from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property

import attrs
import jax
import jax.numpy as jnp
import pandas as pd

from stratified_models.simpler.graph import RegularizationGraph
from stratified_models.simpler.loss import Loss
from stratified_models.simpler.model import Node, StartifiedModel, ThetaShape
from stratified_models.simpler.scalar_function import ScalarFunction


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
        for node, x_slice in self.x.groupby(groupby_cols):
            yield node, x_slice, self.y[x_slice.index]

    def group_losses(self) -> Iterable[tuple[Node, ScalarFunction]]:
        for node, x, y in self.group_data():
            yield (
                node,
                self.loss.build(
                    jnp.asarray(x[self.regression_features].to_numpy()),
                    jnp.asarray(y.to_numpy()),
                ),
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
        compiled_problem = self.compile(abstract_problem)
        model, info = self.solve(compiled_problem, hyperparameters)
        return model, info, compiled_problem
