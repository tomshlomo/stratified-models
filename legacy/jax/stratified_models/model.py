from __future__ import annotations

from collections.abc import Hashable, Sequence
from functools import cached_property
from math import prod

import attrs
import jax
import jax.numpy as jnp
import pandas as pd
import structlog

logger = structlog.get_logger()

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

    @cached_property
    def array_shape(self) -> tuple[int, ...]:
        return *self.graph_sizes, self.m

    @cached_property
    def num_nodes(self) -> int:
        return int(prod(self.graph_sizes))

    def node_to_index(self, node: Node) -> NodeIndex:
        return tuple(
            stratification.get_subnode_index(sub_node)
            for stratification, sub_node in zip(self.stratifications, node, strict=True)
        )


@attrs.frozen(kw_only=True)
class StartifiedModel:
    theta: jax.Array
    shape: ThetaShape

    def theta_at_node(self, node: Node) -> jax.Array:
        return self.theta[self.shape.node_to_index(node)]

    def predict(self, x: pd.DataFrame) -> pd.Series:
        logger.info(
            "model_predict_start",
            rows=x.shape[0],
            num_regression_features=len(self.shape.regression_features),
            num_stratifications=len(self.shape.stratifications),
        )
        indices = tuple(
            stratification.index.get_indexer(x[stratification.name])
            for stratification in self.shape.stratifications
        )
        theta_aligned = self.theta[indices]
        x_reg = jnp.asarray(x.loc[:, self.shape.regression_features].to_numpy())
        y = jnp.sum(x_reg * theta_aligned, axis=1)
        return pd.Series(jax.device_get(y), index=x.index)
