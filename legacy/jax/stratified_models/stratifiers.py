from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence

import attrs
import numpy as np
import pandas as pd
import structlog
from numpy import typing as npt
from sklearn.cluster import KMeans

from stratified_models.graph import (
    NetworkXRegularizationGraph,
    RegularizationGraph,
)

logger = structlog.get_logger()

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
        logger.debug(
            "stratifier_transform_start",
            stratifier=type(self).__name__,
            out_feature_name=str(self.out_feature_name),
            rows=df.shape[0],
        )
        df[self.out_feature_name] = self.transform(df[self.in_feature_names])


class StratifierFitter(ABC):
    @abstractmethod
    def fit(self, x: pd.DataFrame) -> Stratifier:
        pass


@attrs.frozen(kw_only=True)
class BinningStratifierFitter(StratifierFitter, ABC):
    out_feature_name: Hashable

    def fit(
        self,
        x: pd.DataFrame,
    ) -> BinningStratifier:
        feature_name = x.columns[0]
        logger.info(
            "binning_stratifier_fit_start",
            fitter=type(self).__name__,
            feature_name=str(feature_name),
            rows=x.shape[0],
        )
        bin_edges = self.get_bin_edges(x.to_numpy())
        num_bins = len(bin_edges) + 1
        graph = NetworkXRegularizationGraph.path(num_bins, name=self.out_feature_name)
        logger.info(
            "binning_stratifier_fit_complete",
            fitter=type(self).__name__,
            num_bins=num_bins,
            out_feature_name=str(self.out_feature_name),
        )
        return BinningStratifier(
            bin_edges=bin_edges,
            graph=graph,
            in_feature_name=feature_name,
        )

    @abstractmethod
    def get_bin_edges(self, x: Array) -> Array:
        pass


@attrs.frozen(kw_only=True)
class ConstantWidthBinning(BinningStratifierFitter):
    min_width: float
    min_num_bins: int

    def get_bin_edges(self, x: Array) -> Array:
        min_value = x.min()
        max_value = x.max()
        num_bins = math.ceil((max_value - min_value) / self.min_width)
        num_bins = max(num_bins, self.min_num_bins)
        return np.linspace(min_value, max_value, num_bins + 1)[1:-1]


@attrs.frozen(kw_only=True)
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
        logger.debug(
            "binning_stratifier_transform_start",
            out_feature_name=str(self.out_feature_name),
            num_bins=len(self.bin_edges) + 1,
            rows=x.shape[0],
        )
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
        logger.info(
            "kmeans_stratifier_fit_start",
            n_samples=x.shape[0],
            n_features=x.shape[1],
            n_clusters=self.kmeans.n_clusters,
            out_feature_name=str(self.out_feature_name),
        )
        self.kmeans.fit(x)
        graph = NetworkXRegularizationGraph.voronoi(
            self.kmeans.cluster_centers_,
            name=self.out_feature_name,
        )
        logger.info(
            "kmeans_stratifier_fit_complete",
            n_clusters=self.kmeans.n_clusters,
            out_feature_name=str(self.out_feature_name),
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
        logger.debug(
            "kmeans_stratifier_transform_start",
            out_feature_name=str(self.out_feature_name),
            n_clusters=self.kmeans.n_clusters,
            rows=x.shape[0],
        )
        return pd.Series(
            self.kmeans.predict(x),
            index=x.index,
            name=self.out_feature_name,
        )
