from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

from stratified_models.simpler.graph import RegularizationGraph
from stratified_models.simpler.stratifiers import (
    BinningStratifier as FunctionalBinningStratifier,
    ConstantWidth,
    KMeansConfig,
    KMeansStratifier as FunctionalKMeansStratifier,
    Quantiles,
)

_DEFAULT_FEATURE_NAME = "stratification_feature"


def _as_series(x: pd.DataFrame) -> pd.Series:
    if x.shape[1] == 1:
        return x.iloc[:, 0]
    return pd.Series(
        x.to_numpy().tolist(),
        index=x.index,
        name=_DEFAULT_FEATURE_NAME,
    )


class BinningStratifier(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        bin_width: float | Sequence[float] | None,
        n_bins: int | Sequence[int] | None,
        strategy: Literal["uniform", "quantile"],
    ) -> None:
        self.bin_width = bin_width
        self.n_bins = n_bins
        self.strategy = strategy
        self._impl: FunctionalBinningStratifier | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: None = None,  # noqa: ARG002
    ) -> BinningStratifier:
        self._impl = self._build_strategy().fit(_as_series(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["_impl"])
        assert self._impl is not None
        return self._impl.transform(_as_series(X)).to_frame()

    @property
    def graph(self) -> RegularizationGraph:
        check_is_fitted(self, ["_impl"])
        assert self._impl is not None
        return self._impl.graph

    def _build_strategy(self) -> ConstantWidth | Quantiles:
        if self.strategy == "uniform":
            if self.bin_width is None:
                msg = "bin_width must be specified for uniform strategy."
                raise ValueError(msg)
            return ConstantWidth(bin_width=self.bin_width)
        if self.strategy == "quantile":
            if self.n_bins is None:
                msg = "n_bins must be specified for quantile strategy."
                raise ValueError(msg)
            return Quantiles(n_bins=self.n_bins)
        msg = f"Unknown strategy: {self.strategy}"
        raise ValueError(msg)


class KMeansStratifier(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        n_clusters: int | None,
        random_state: int | None,
        n_init: int | Literal["auto"],
        kmeans: KMeans | None,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.kmeans = kmeans
        self._impl: FunctionalKMeansStratifier | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: None = None,  # noqa: ARG002
    ) -> KMeansStratifier:
        if self.kmeans is not None:
            self._impl = FunctionalKMeansStratifier.from_kmeans(
                _as_series(X),
                kmeans=self.kmeans,
            )
            return self
        if self.n_clusters is None:
            msg = "n_clusters or kmeans must be specified."
            raise ValueError(msg)
        config = KMeansConfig(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self._impl = FunctionalKMeansStratifier.fit(
            _as_series(X),
            config=config,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["_impl"])
        assert self._impl is not None
        return self._impl.transform(_as_series(X)).to_frame()

    @property
    def graph(self) -> RegularizationGraph:
        check_is_fitted(self, ["_impl"])
        assert self._impl is not None
        return self._impl.graph
