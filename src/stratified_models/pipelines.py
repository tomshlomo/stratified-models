from __future__ import annotations

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
    transformers: TransformPipeline
    model: StartifiedModel

    def transform_and_predict(self, x: pd.DataFrame) -> pd.Series:
        logger.info(
            "pipeline_predict_start",
            rows=x.shape[0],
            num_stratifiers=len(self.transformers.stratifiers),
            normalize=self.transformers.normalizer is not None,
        )
        self.transformers.transform(x)
        y_pred = self.model.predict(x)
        return self.transformers.inverse_transform_target(y_pred)


@attrs.frozen(kw_only=True)
class TransformPipeline:
    stratifiers: Sequence[Stratifier]
    normalizer: StandardScaler | None
    target_normalizer: StandardScaler | None
    regression_features: Sequence[str]
    intercept: bool

    def transform(self, df: pd.DataFrame) -> None:
        for stratifier in self.stratifiers:
            stratifier.transform_df_inplace(df)
        if self.normalizer is not None:
            df[self.regression_features] = self.normalizer.transform(
                df[self.regression_features]
            )
            logger.debug(
                "pipeline_transform_normalized",
                rows=df.shape[0],
                num_regression_features=len(self.regression_features),
            )
        if self.intercept:
            df["one"] = 1.0
            logger.debug("pipeline_transform_intercept_added", rows=df.shape[0])

    def transform_target(self, y: pd.Series) -> pd.Series:
        if self.target_normalizer is None:
            return y
        y_vec = y.to_numpy().reshape(-1, 1)
        y_transformed = self.target_normalizer.transform(y_vec).flatten()
        return pd.Series(y_transformed, index=y.index, name=y.name)

    def inverse_transform_target(self, y: pd.Series) -> pd.Series:
        if self.target_normalizer is None:
            return y
        y_vec = y.to_numpy().reshape(-1, 1)
        y_original = self.target_normalizer.inverse_transform(y_vec).flatten()
        return pd.Series(y_original, index=y.index, name=y.name)


@attrs.frozen(kw_only=True)
class CompiledPipelineFitProblem[T]:
    transformers: TransformPipeline
    solver_compiled_problem: T


@attrs.frozen(kw_only=True)
class StratifiedPipelineFitter[T, I: SolveInfo]:
    stratifiers_fitters: Sequence[tuple[Sequence[Hashable], StratifierFitter]] = ()
    graphs: Sequence[RegularizationGraph] = ()
    normalize: bool = True
    normalize_target: bool = True
    regression_features: Sequence[str] = ()
    intercept: bool = True
    solver: Solver[T, I]
    loss: Loss = attrs.field(factory=SumOfSquaresLoss)
    regularizers: Mapping[str, ScalarFunction] = attrs.field(factory=dict)

    def _fit_transformers(self, x: pd.DataFrame, y: pd.Series) -> TransformPipeline:
        logger.info(
            "pipeline_fit_transformers_start",
            rows=x.shape[0],
            num_stratifiers=len(self.stratifiers_fitters),
            normalize=self.normalize,
            normalize_target=self.normalize_target,
        )
        stratifiers = []
        for feature_names, stratifier_fitter in self.stratifiers_fitters:
            logger.info(
                "pipeline_stratifier_fit_start",
                stratifier_fitter=type(stratifier_fitter).__name__,
                num_features=len(feature_names),
            )
            stratifier = stratifier_fitter.fit(x[feature_names])
            stratifiers.append(stratifier)
            logger.info(
                "pipeline_stratifier_fit_complete",
                stratifier=type(stratifier).__name__,
                out_feature_name=str(stratifier.out_feature_name),
            )
        if self.normalize:
            normalizer = StandardScaler()
            normalizer.fit(x[self.regression_features])
            logger.debug(
                "pipeline_fit_normalized",
                rows=x.shape[0],
                num_regression_features=len(self.regression_features),
            )
        else:
            normalizer = None

        if self.normalize_target:
            target_normalizer = StandardScaler()
            target_normalizer.fit(y.to_numpy().reshape(-1, 1))
            logger.debug("pipeline_fit_target_normalized", rows=y.shape[0])
        else:
            target_normalizer = None

        return TransformPipeline(
            stratifiers=stratifiers,
            normalizer=normalizer,
            target_normalizer=target_normalizer,
            regression_features=self.regression_features,
            intercept=self.intercept,
        )

    def _build_solver_abstract_problem(
        self,
        transformers: TransformPipeline,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> AbstractProblem:
        logger.info("pipeline_build_solver_problem_start", rows=x.shape[0])
        problem_regression_features = list(transformers.regression_features)
        if transformers.intercept:
            problem_regression_features.append("one")

        y_transformed = transformers.transform_target(y)

        return AbstractProblem(
            x=x,
            y=y_transformed,
            regression_features=problem_regression_features,
            graphs=[s.graph for s in transformers.stratifiers] + list(self.graphs),
            regularizers=self.regularizers,
            loss=self.loss,
        )

    def compile(self, x: pd.DataFrame, y: pd.Series) -> CompiledPipelineFitProblem[T]:
        transformers = self._fit_transformers(x, y)
        transformers.transform(x)
        solver_abstract_problem = self._build_solver_abstract_problem(
            transformers, x, y
        )

        logger.info(
            "pipeline_compile_start",
            solver=type(self.solver).__name__,
            rows=solver_abstract_problem.n,
            num_graphs=len(solver_abstract_problem.graphs),
            num_regularizers=len(solver_abstract_problem.regularizers),
        )
        solver_compiled_problem = self.solver.compile(solver_abstract_problem)
        logger.info("pipeline_compile_complete", solver=type(self.solver).__name__)

        return CompiledPipelineFitProblem(
            transformers=transformers,
            solver_compiled_problem=solver_compiled_problem,
        )

    def fit(
        self,
        compiled: CompiledPipelineFitProblem[T],
        hyperparameters: Hyperparameters,
    ) -> tuple[StratifiedPipeline, I]:
        logger.info("pipeline_solve_start", solver=type(self.solver).__name__)
        model, info = self.solver.solve(
            compiled.solver_compiled_problem, hyperparameters
        )
        logger.info(
            "pipeline_solve_complete",
            solver=type(self.solver).__name__,
            converged=info.converged(),
            solve_info=type(info).__name__,
        )
        return (
            StratifiedPipeline(
                transformers=compiled.transformers,
                model=model,
            ),
            info,
        )
