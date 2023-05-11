from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Optional, Union

import numpy as np
import pandas as pd

from stratified_models.fitters.fitter import Fitter, ProblemUpdate, RefitDataType
from stratified_models.losses import LossFactory
from stratified_models.problem import F, StratifiedLinearRegressionProblem, Theta
from stratified_models.regularization_graph.regularization_graph import (
    RegularizationGraph,
)


@dataclass
class EstimatorPreviousFitData(Generic[F, RefitDataType]):
    loss_factory: LossFactory[F]
    regularizers: list[tuple[F, float]]
    graphs: list[tuple[RegularizationGraph[F], float]]
    regression_features: list[str]
    fitter: Fitter[F, RefitDataType]
    x: pd.DataFrame
    y: pd.Series
    fitter_refit_data: RefitDataType
    theta: Theta


@dataclass
class StratifiedLinearEstimator(Generic[F, RefitDataType]):
    loss_factory: LossFactory[F]
    regularizers: list[tuple[F, float]]
    graphs: list[tuple[RegularizationGraph[F], float]]
    regression_features: list[str]
    fitter: Fitter[F, RefitDataType]
    warm_start: bool

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> StratifiedLinearEstimator[F, RefitDataType]:
        theta, fitter_refit_data = self.attempt_refit(x=X, y=y)
        if theta is None or fitter_refit_data is None:
            theta, fitter_refit_data = self._fit_from_scratch(x=X, y=y)
        self.previous_fit_data_ = self._build_previous_fit_data(
            x=X,
            y=y,
            fitter_refit_data=fitter_refit_data,
            theta=theta,
        )
        self.theta_ = theta
        return self

    def attempt_refit(
        self, x: pd.DataFrame, y: pd.Series
    ) -> Union[tuple[Theta, RefitDataType], tuple[None, None]]:
        previous_fit_data_ = self._get_previous_fit_data()
        if not previous_fit_data_:
            return None, None
        problem_update = self._get_problem_update(
            previous_fit_data=previous_fit_data_, x=x, y=y
        )
        if not problem_update:
            return None, None
        theta, fitter_refit_data, _ = self.fitter.refit(
            problem_update=problem_update,
            refit_data=previous_fit_data_.fitter_refit_data,
        )
        return theta, fitter_refit_data

    def _fit_from_scratch(
        self, x: pd.DataFrame, y: pd.Series
    ) -> tuple[Theta, RefitDataType]:
        problem = self._get_problem(x=x, y=y)
        theta, fitter_refit_data, _ = self.fitter.fit(problem=problem)
        return theta, fitter_refit_data

    def _build_previous_fit_data(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        fitter_refit_data: RefitDataType,
        theta: Theta,
    ) -> EstimatorPreviousFitData[F, RefitDataType]:
        return EstimatorPreviousFitData(
            loss_factory=self.loss_factory,
            regularizers=self.regularizers,
            graphs=self.graphs,
            regression_features=self.regression_features,
            fitter=self.fitter,
            x=x,
            y=y,
            fitter_refit_data=fitter_refit_data,
            theta=theta,
        )

    def _get_problem(
        self, x: pd.DataFrame, y: pd.Series
    ) -> StratifiedLinearRegressionProblem[F]:
        return StratifiedLinearRegressionProblem(
            x=x,
            y=y,
            loss_factory=self.loss_factory,
            regularizers=self.regularizers,
            graphs=self.graphs,
            regression_features=self.regression_features,
        )

    def _get_problem_update(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        previous_fit_data: EstimatorPreviousFitData[F, RefitDataType],
    ) -> Optional[ProblemUpdate]:
        # check if structure changed
        if not (
            [f for f, _ in self.regularizers]
            == [f for f, _ in previous_fit_data.regularizers]
            and [f for f, _ in self.graphs] == [f for f, _ in previous_fit_data.graphs]
            and self.loss_factory == previous_fit_data.loss_factory
            and self.regression_features == previous_fit_data.regression_features
        ):
            return None

        # check if fitter changed
        if self.fitter != previous_fit_data.fitter:
            # todo: in this case, we should at least use the given theta as a hint for
            #  next fit
            return None

        # check if data changed
        if not (x is previous_fit_data.x and y is previous_fit_data.y):
            # todo: in this case, we should at least use the given theta as a hint for
            #  next fit
            return None

        return ProblemUpdate(
            new_graph_gammas=[gamma for _, gamma in self.graphs],
            new_regularization_gammas=[gamma for _, gamma in self.regularizers],
        )

    def _get_previous_fit_data(
        self,
    ) -> Optional[EstimatorPreviousFitData[F, RefitDataType]]:
        if not self.warm_start:
            return None
        return getattr(self, "previous_fit_data_", None)

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
