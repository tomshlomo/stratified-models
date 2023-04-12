from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Optional

import numpy as np
import pandas as pd

from stratified_models.fitters.fitter import Fitter, RefitDataType
from stratified_models.losses import LossFactory
from stratified_models.problem import F, StratifiedLinearRegressionProblem
from stratified_models.regularization_graph.regularization_graph import (
    RegularizationGraph,
)


@dataclass
class StratifiedLinearEstimator(Generic[F, RefitDataType]):
    loss_factory: LossFactory[F]
    regularizers: list[tuple[F, float]]
    graphs: list[tuple[RegularizationGraph[F], float]]
    regression_features: list[str]
    fitter: Fitter[F, RefitDataType]
    warm_start: bool

    def fit(self, X, y) -> StratifiedLinearEstimator:
        problem = StratifiedLinearRegressionProblem(
            x=X,
            y=y,
            loss_factory=self.loss_factory,
            regularizers=self.regularizers,
            graphs=self.graphs,
            regression_features=self.regression_features,
        )
        refit_data = self._get_refit_data()
        self.theta_, self.refit_data_ = self.fitter.fit(problem, refit_data=refit_data)
        return self

    def _get_refit_data(self) -> Optional[RefitDataType]:
        if not self.warm_start:
            return
        return getattr(self, "refit_data_", None)

    def stratification_features(self) -> list[Hashable]:
        return [graph.name() for graph, _ in self.graphs]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        rows = pd.MultiIndex.from_arrays(
            X.loc[:, self.stratification_features()].values.T
        )
        theta_aligned = self.theta_.df.loc[rows, :]
        y = np.einsum(
            "nm,nm->n",
            X.loc[:, self.regression_features].values,
            theta_aligned.values,
        )
        return pd.Series(y, index=X.index)
