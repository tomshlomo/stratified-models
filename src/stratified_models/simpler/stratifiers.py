from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence

import attrs
import numpy as np
import pandas as pd
from numpy import typing as npt
from sklearn.cluster import KMeans

from stratified_models.simpler.graph import (
    NetworkXRegularizationGraph,
    RegularizationGraph,
)

logger = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]
_DEFAULT_FEATURE_NAME = "stratification_feature"
_EXPECTED_NDIMS = 2


class Stratifier(ABC):
    in_feature_names: Sequence[Hashable]
    graph: RegularizationGraph

    @abstractmethod
    def transform(self, x: pd.DataFrame) -> pd.Series:
        pass

    @property
    def out_feature_name(self) -> Hashable:
        return self.graph.stratification.name

    def transform_df_inplace(self, df: pd.DataFrame) -> None:
        df[self.out_feature_name] = self.transform(df[self.in_feature_names])


class StratifierFitter(ABC):
    @abstractmethod
    def fit(self, x: pd.DataFrame) -> Stratifier:
        pass

    def fit_transform_df_in_place(self, x: pd.DataFrame) -> Stratifier:
        stratifier = self.fit(x)
        stratifier.transform_df_inplace(x)
        return stratifier


@attrs.frozen(kw_only=True)
class BinningStratifierFitter(StratifierFitter, ABC):
    out_feature_name: Hashable

    def fit(
        self,
        x: pd.DataFrame,
    ) -> BinningStratifier:
        bin_edges = self.get_bin_edges(x.to_numpy())
        num_bins = len(bin_edges) + 1
        graph = NetworkXRegularizationGraph.path(num_bins, name=self.out_feature_name)
        return BinningStratifier(
            bin_edges=bin_edges, graph=graph, in_feature_name=x.name
        )

    @abstractmethod
    def get_bin_edges(self, x: Array) -> Array:
        pass


class ConstantWidthBinning(BinningStratifierFitter):
    min_width: float
    min_num_bins: int

    def get_bin_edges(self, x: Array) -> Array:
        min_value = x.min()
        max_value = x.max()
        num_bins = math.ceil((max_value - min_value) / self.min_width)
        num_bins = max(num_bins, self.min_num_bins)
        return np.linspace(min_value, max_value, num_bins + 1)[1:-1]


class QuantilesBinning(BinningStratifierFitter):
    n_bins: int

    def get_bin_edges(self, x: Array) -> Array:
        return np.quantile(x, np.linspace(0, 1, self.n_bins + 1))[1:-1]


@attrs.frozen(kw_only=True)
class BinningStratifier(Stratifier):
    bin_edges: Array
    graph: RegularizationGraph
    in_feature_name: Hashable

    @property
    def in_feature_names(self) -> Sequence[Hashable]:
        return [self.in_feature_name]

    def transform(self, x: pd.DataFrame) -> pd.Series:
        return pd.Series(
            np.digitize(x[self.in_feature_name].to_numpy(), self.bin_edges),
            index=x.index,
            name=self.out_feature_name,
        )


@attrs.frozen(kw_only=True)
class KMeansStratifierFitter(StratifierFitter):
    out_feature_name: Hashable
    kmeans: KMeans

    def fit(
        self,
        x: pd.DataFrame,
    ) -> KMeansStratifier:
        self.kmeans.fit(x)
        graph = NetworkXRegularizationGraph.voronoi(
            self.kmeans.cluster_centers_,
            name=self.out_feature_name,
        )
        return KMeansStratifier(
            kmeans=self.kmeans,
            graph=graph,
            in_feature_names=list(x.columns),
        )


@attrs.frozen(kw_only=True)
class KMeansStratifier(Stratifier):
    kmeans: KMeans
    graph: RegularizationGraph
    in_feature_names: Sequence[Hashable]

    def transform(self, x: pd.DataFrame) -> pd.Series:
        x = x.loc[:, self.in_feature_names]
        return pd.Series(
            self.kmeans.predict(x),
            index=x.index,
            name=self.out_feature_name,
        )
