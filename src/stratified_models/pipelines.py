from collections.abc import Hashable, Mapping, Sequence

import attrs
import pandas as pd
import structlog
from sklearn.preprocessing import StandardScaler

from stratified_models.graph import RegularizationGraph
from stratified_models.loss import Loss, SumOfSquaresLoss
from stratified_models.model import StartifiedModel
from stratified_models.scalar_function import ScalarFunction
from stratified_models.solvers import (
    AbstractProblem,
    Hyperparameters,
    SolveInfo,
    Solver,
)
from stratified_models.stratifiers import (
    Stratifier,
    StratifierFitter,
)

logger = structlog.get_logger()


@attrs.frozen(kw_only=True)
class StratifiedPipeline:
    stratifiers: Sequence[Stratifier]
    normalizer: StandardScaler | None
    model: StartifiedModel

    def predict(self, x: pd.DataFrame) -> pd.Series:
        logger.info(
            "pipeline_predict_start",
            rows=x.shape[0],
            num_stratifiers=len(self.stratifiers),
            normalize=self.normalizer is not None,
        )
        for stratifier in self.stratifiers:
            stratifier.transform_df_inplace(x)
        if self.normalizer is not None:
            x[self.model.shape.regression_features] = self.normalizer.transform(
                x[self.model.shape.regression_features]
            )
            logger.debug(
                "pipeline_predict_normalized",
                rows=x.shape[0],
                num_regression_features=len(self.model.shape.regression_features),
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
        logger.info(
            "pipeline_fit_start",
            rows=x.shape[0],
            num_regression_features=len(self.regression_features),
            num_stratifiers=len(self.stratifiers_fitters),
            num_graphs=len(self.graphs),
            normalize=self.normalize,
            intercept=self.intercept,
        )
        stratifiers = []
        for feature_names, stratifier_fitter in self.stratifiers_fitters:
            logger.info(
                "pipeline_stratifier_fit_start",
                stratifier_fitter=type(stratifier_fitter).__name__,
                num_features=len(feature_names),
            )
            stratifier = stratifier_fitter.fit(x[feature_names])
            stratifier.transform_df_inplace(x)
            stratifiers.append(stratifier)
            logger.info(
                "pipeline_stratifier_fit_complete",
                stratifier=type(stratifier).__name__,
                out_feature_name=str(stratifier.out_feature_name),
            )
        if self.normalize:
            normalizer = StandardScaler()
            x[self.regression_features] = normalizer.fit_transform(
                x[self.regression_features]
            )
            logger.debug(
                "pipeline_fit_normalized",
                rows=x.shape[0],
                num_regression_features=len(self.regression_features),
            )
        else:
            normalizer = None
        if self.intercept:
            x["one"] = 1.0
            logger.debug("pipeline_fit_intercept_added", rows=x.shape[0])
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
        logger.info(
            "pipeline_fit_complete",
            converged=info.converged(),
            solve_info=type(info).__name__,
        )
        return (
            StratifiedPipeline(
                stratifiers=stratifiers,
                normalizer=normalizer,
                model=model,
            ),
            info,
        )
