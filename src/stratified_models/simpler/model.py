from __future__ import annotations

from collections.abc import Hashable, Sequence
from functools import cached_property

import attrs
import numpy as np
import pandas as pd

from stratified_models.linear_operator import Array

Node = tuple[Hashable, ...]
NodeIndex = tuple[int, ...]


@attrs.frozen(kw_only=True)
class Stratification:
    index: pd.Index

    @property
    def name(self) -> Hashable:
        return self.index.name

    @property
    def size(self) -> int:
        return self.index.shape[0]

    def get_subnode_index(self, sub_node: Hashable) -> int:
        return self.index.get_loc(sub_node)


@attrs.frozen(kw_only=True)
class ThetaShape:
    regression_features: Sequence[str]
    stratifications: Sequence[Stratification]

    @property
    def m(self) -> int:
        return len(self.regression_features)

    @cached_property
    def stratification_features(self) -> Sequence[Hashable]:
        return tuple(stratification.name for stratification in self.stratifications)

    @cached_property
    def graph_sizes(self) -> tuple[int, ...]:
        return tuple(stratification.size for stratification in self.stratifications)

    # @cached_property
    # def dims(self) -> tuple[int, ...]:
    #     # return self.m, *self.graph_sizes
    #     return self.m, *self.graph_sizes

    @cached_property
    def array_shape(self) -> tuple[int, ...]:
        return *self.graph_sizes, self.m

    @cached_property
    def num_nodes(self) -> int:
        return int(np.prod(self.graph_sizes))

    # @cached_property
    # def flat_shape(self) -> tuple[int, int]:  # TODO: remove?
    #     return self.num_nodes, self.m

    # def node_to_flat_index(self, node: Node) -> int:
    #     return self.index_to_flat_index(self.node_to_index(node))

    # def index_to_flat_index(self, index: tuple[int, ...]) -> int:
    #     return int(np.ravel_multi_index(index, self.graph_sizes))

    def node_to_index(self, node: Node) -> NodeIndex:
        return tuple(
            stratification.get_subnode_index(sub_node)
            for stratification, sub_node in zip(self.stratifications, node, strict=True)
        )

    # def to_pandas_multi_index(self) -> pd.MultiIndex:
    #     return pd.MultiIndex.from_product(
    #         [stratification.index for stratification in self.stratifications]
    #     )


@attrs.frozen(kw_only=True)
class StartifiedModel:
    # theta: pd.DataFrame
    theta: Array
    shape: ThetaShape

    def theta_at_node(self, node: Node) -> Array:
        return self.theta[self.shape.node_to_index(node)]

    # @property
    # def stratification_features(self) -> Sequence[Hashable]:
    #     return self.shape.stratification_features

    # @property
    # def regression_features(self) -> pd.Index:
    #     return self.shape.regression_features

    # def as_flat_numpy_array(self) -> Array:
    #     """Return theta as a `(num_nodes, m)` array.

    #     Row order matches `ThetaShape.index_to_flat_index` (C-order over graph sizes).
    #     """
    #     return self.theta.to_numpy()

    # def as_numpy_array(self) -> Array:
    #     """Return theta as a tensor shaped `(m, graph_size0, graph_size1, ...)`."""
    #     flat = self.as_flat_numpy_array()
    #     return flat.T.reshape(self.shape.dims, order="C")

    def predict(self, x: pd.DataFrame) -> pd.Series:
        # rows = pd.MultiIndex.from_arrays(
        #     x.loc[:, self.stratification_features].to_numpy().T,
        # )
        indices = np.array(
            [s.index.get_loc(x[s.name]) for s in self.shape.stratifications]
        )
        theta_aligned = self.theta[indices]
        y = np.einsum(
            "nm,nm->n",
            x.loc[:, self.shape.regression_features].to_numpy(),
            theta_aligned,
        )
        return pd.Series(y, index=x.index)

    # @staticmethod
    # def from_array(arr: Array, shape: ThetaShape) -> StartifiedModel:
    #     theta = pd.DataFrame(
    #         arr.reshape(shape.flat_shape),
    #         index=shape.to_pandas_multi_index(),
    #         columns=shape.regression_features,  # ty:ignore[invalid-argument-type]
    #     )
    #     return StartifiedModel(theta=theta, shape=shape)
