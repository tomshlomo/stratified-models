from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd

from stratified_models.losses import LossFactory
from stratified_models.regularization_graph.regularization_graph import (
    RegularizationGraph,
)
from stratified_models.regularizers import RegularizationFactory
from stratified_models.scalar_function import Array, ScalarFunction

F = TypeVar("F", bound=ScalarFunction[Array])


@dataclass
class StratifiedLinearRegressionProblem(Generic[F]):
    x: pd.DataFrame
    y: pd.Series
    loss_factory: LossFactory[F]
    regularizers_factories: tuple[tuple[RegularizationFactory[F], float], ...]
    graphs: tuple[tuple[RegularizationGraph[F], float], ...]
    regression_features: list[str]

    @property
    def m(self) -> int:
        return len(self.regression_features)

    @property
    def n(self) -> int:
        return self.x.shape[0]  # type:ignore[no-any-return]

    @property
    def stratification_features(self) -> list[Hashable]:
        return [graph.name() for graph, _ in self.graphs]

    def loss_iter(self) -> Iterable[tuple[F, tuple[Hashable, ...]]]:
        for node, x, y in self.node_data_iter():
            yield self.loss_factory.build_loss_function(
                x[self.regression_features].values, y.values
            ), node

    def node_data_iter(
        self,
    ) -> Iterable[tuple[tuple[Hashable, ...], pd.DataFrame, pd.Series,]]:
        for node, x_slice in self.x.groupby(self.stratification_features)[
            self.regression_features
        ]:
            if not isinstance(node, tuple):
                node = (node,)
            yield node, x_slice, self.y.loc[x_slice.index]

    def laplacians(self) -> Iterable[tuple[F, float]]:
        dims = self.theta_shape()
        for i, (graph, gamma) in enumerate(self.graphs):
            yield graph.laplacian(axis=i, dims=dims), gamma

    def theta_shape(self) -> tuple[int, ...]:  # todo: remove?
        return *self.graph_sizes(), self.m

    def graph_sizes(self) -> tuple[int, ...]:
        return tuple(graph.number_of_nodes() for graph, _ in self.graphs)

    def theta_flat_shape(self) -> tuple[int, int]:  # todo: remove?
        k = int(np.prod(self.graph_sizes()))
        return k, self.m

    def get_node_flat_index(self, node: tuple[Hashable, ...]) -> int:
        return self.to_flat_index(self.get_node_index(node))

    def to_flat_index(self, index: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(index, self.graph_sizes()))

    def get_node_index(self, node: tuple[Hashable, ...]) -> tuple[int, ...]:
        return tuple(
            graph.get_node_index(sub_node)
            for (graph, _), sub_node in zip(self.graphs, node)
        )

    def regularizers(self) -> Iterable[tuple[F, float]]:
        shape = self.theta_shape()
        for factory, gamma in self.regularizers_factories:
            yield factory.build_regularization_function(shape), gamma

    def cost(self, theta: Theta) -> float:
        cost = 0.0
        for loss, node in self.loss_iter():
            cost += loss(theta.df.loc[node].values)
        for reg, gamma in self.regularizers():
            cost += gamma * reg(theta.df.values)
        for lap, gamma in self.laplacians():
            cost += gamma * lap(theta.as_array())
        return cost


@dataclass
class Theta:
    df: pd.DataFrame
    shape: tuple[int, ...]

    def as_array(self) -> Array:
        return self.df.values.reshape(self.shape)  # type:ignore[no-any-return]

    @classmethod
    def _get_df_from_array(
        cls,
        arr: Array,
        problem: StratifiedLinearRegressionProblem[F],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            arr.reshape(problem.theta_flat_shape()),
            index=pd.MultiIndex.from_product(
                graph.nodes for graph, _ in problem.graphs
            ),
            columns=problem.regression_features,
        )

    @classmethod
    def from_array(
        cls,
        arr: Array,
        problem: StratifiedLinearRegressionProblem[F],
    ) -> Theta:
        df = cls._get_df_from_array(arr=arr, problem=problem)
        return Theta(df=df, shape=problem.theta_shape())

    def stratification_features(self) -> Sequence[Hashable]:
        return self.df.index.names  # type:ignore[no-any-return]

    def regression_features(self) -> pd.Index:
        return self.df.columns

    def predict(self, x: pd.DataFrame) -> pd.Series:
        rows = pd.MultiIndex.from_arrays(
            x.loc[:, self.stratification_features()].values.T
        )
        theta_aligned = self.df.loc[rows, :]
        y = np.einsum(
            "nm,nm->n",
            x.loc[:, self.regression_features()].values,
            theta_aligned.values,
        )
        return pd.Series(y, index=x.index)

    # def get_local(self, index: tuple[Hashable, ...]) -> pd.DataFrame:
    #     return self.df.xs(key=index, axis=0)
    #
    # def as_dict(self) -> dict[tuple[Hashable, ...], pd.Series]:
    #     return dict(self.df.iterrows())
