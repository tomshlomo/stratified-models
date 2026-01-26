from collections.abc import Hashable, Mapping, Sequence

import attrs
import pandas as pd
from sklearn.preprocessing import StandardScaler

from stratified_models.simpler.graph import RegularizationGraph
from stratified_models.simpler.loss import Loss, SumOfSquaresLoss
from stratified_models.simpler.model import StartifiedModel
from stratified_models.simpler.scalar_function import ScalarFunction
from stratified_models.simpler.solvers import (
    AbstractProblem,
    Hyperparameters,
    SolveInfo,
    Solver,
)
from stratified_models.simpler.stratifiers import (
    Stratifier,
    StratifierFitter,
)


@attrs.frozen(kw_only=True)
class StratifiedPipeline:
    stratifiers: Sequence[Stratifier]
    normalizer: StandardScaler | None
    model: StartifiedModel

    def predict(self, x: pd.DataFrame) -> pd.Series:
        for stratifier in self.stratifiers:
            stratifier.transform_df_inplace(x)
        if self.normalizer is not None:
            x[self.model.shape.regression_features] = self.normalizer.transform(
                x[self.model.shape.regression_features]
            )
        return self.model.predict(x)


@attrs.frozen(kw_only=True)
class StratifiedPipelineFitter[T, I: SolveInfo]:
    stratifiers_fitters: Sequence[tuple[Sequence[Hashable], StratifierFitter]] = ()
    graphs: Sequence[RegularizationGraph] = ()
    normalize: bool = True
    regression_features: Sequence[str] = ()
    intercept: bool = True
    solver: Solver[T, I]
    loss: Loss = attrs.field(factory=SumOfSquaresLoss)
    regularizers: Mapping[str, ScalarFunction] = attrs.field(factory=dict)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        hyperparameters: Hyperparameters,
    ) -> tuple[StratifiedPipeline, I]:
        stratifiers = []
        for feature_names, stratifier_fitter in self.stratifiers_fitters:
            stratifier = stratifier_fitter.fit_transform_df_in_place(x[feature_names])
            stratifiers.append(stratifier)
        if self.normalize:
            normalizer = StandardScaler()
            x[self.regression_features] = normalizer.fit_transform(
                x[self.regression_features]
            )
        else:
            normalizer = None
        if self.intercept:
            x["one"] = 1.0
        abstract_problem = AbstractProblem(
            x=x,
            y=y,
            regression_features=self.regression_features,
            graphs=[stratifier.graph for stratifier in stratifiers] + list(self.graphs),
            regularizers=self.regularizers,
            loss=self.loss,
        )
        model, info, _ = self.solver.compile_and_solve(
            abstract_problem, hyperparameters
        )
        return (
            StratifiedPipeline(
                stratifiers=stratifiers,
                normalizer=normalizer,
                model=model,
            ),
            info,
        )
