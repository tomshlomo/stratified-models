from collections.abc import Callable, Mapping, Sequence

import attrs
import numpy as np
import optuna
import pandas as pd
import structlog
from numpy import typing as npt
from sklearn.metrics import r2_score
from sklearn.model_selection import BaseCrossValidator, KFold

from stratified_models.pipelines import (
    CompiledPipelineFitProblem,
    StratifiedPipeline,
    StratifiedPipelineFitter,
)
from stratified_models.solvers import Hyperparameters, SolveInfo

logger = structlog.get_logger()

Scorer = Callable[[npt.ArrayLike, npt.ArrayLike], float]
Aggregator = Callable[[Sequence[float]], float]


@attrs.frozen(kw_only=True)
class TuningResult[I: SolveInfo]:
    best_hyperparameters: Hyperparameters
    best_pipeline: StratifiedPipeline
    best_solve_info: I
    study: optuna.Study


@attrs.frozen(kw_only=True)
class OptunaTuner[T, I: SolveInfo]:
    pipeline_fitter: StratifiedPipelineFitter[T, I]
    regularizers_range: Mapping[str, tuple[float, float]]
    graphs_range: Mapping[str, tuple[float, float]]

    cv: BaseCrossValidator = attrs.field(
        factory=lambda: KFold(n_splits=5, shuffle=True)
    )
    scorer: Scorer = r2_score
    aggregator: Aggregator = attrs.field(factory=lambda: np.mean)
    sampler: optuna.samplers.BaseSampler = attrs.field(
        factory=lambda: optuna.samplers.TPESampler(seed=42, n_startup_trials=5)
    )

    n_trials: int = 50
    timeout: float | None = None
    reduction_factor: int = 3

    def tune(self, x: pd.DataFrame, y: pd.Series) -> TuningResult[I]:
        n_rows = len(x)
        default_value = 1e-4 * n_rows

        # Pre-compile all folds
        compiled_folds = self._compile_folds(x, y)
        n_folds = len(compiled_folds)

        def objective(trial: optuna.Trial) -> float:
            hyperparams = self._suggest_hyperparams(trial)
            fold_scores = self._evaluate_folds(trial, compiled_folds, hyperparams)
            return self.aggregator(fold_scores)

        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=n_folds,
            reduction_factor=self.reduction_factor,
        )
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            sampler=self.sampler,
        )

        # Enqueue initial trial with default values
        initial_params = self._build_initial_params(default_value)
        study.enqueue_trial(initial_params)

        logger.info(
            "tuning_optimize_start",
            n_trials=self.n_trials,
            timeout=self.timeout,
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )
        logger.info(
            "tuning_optimize_complete",
            best_value=study.best_value,
            n_trials=len(study.trials),
        )

        best_hyperparams = self._extract_best_hyperparams(study)

        # Final fit on full data
        logger.info("tuning_final_fit_start")
        final_compiled = self.pipeline_fitter.compile(x=x.copy(), y=y)
        best_pipeline, best_info = self.pipeline_fitter.fit(
            compiled=final_compiled,
            hyperparameters=best_hyperparams,
        )
        logger.info("tuning_final_fit_complete", converged=best_info.converged())

        return TuningResult(
            best_hyperparameters=best_hyperparams,
            best_pipeline=best_pipeline,
            best_solve_info=best_info,
            study=study,
        )

    def _compile_folds(
        self, x: pd.DataFrame, y: pd.Series
    ) -> list[tuple[CompiledPipelineFitProblem[T], pd.DataFrame, pd.Series]]:
        compiled_folds: list[
            tuple[CompiledPipelineFitProblem[T], pd.DataFrame, pd.Series]
        ] = []

        logger.info(
            "tuning_compile_folds_start",
            n_folds=self.cv.get_n_splits(x),
            n_rows=len(x),
        )
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(x, y)):
            fold_train_x = x.iloc[train_idx].copy()
            fold_train_y = y.iloc[train_idx]
            fold_val_x = x.iloc[val_idx].copy()
            fold_val_y = y.iloc[val_idx]

            compiled = self.pipeline_fitter.compile(x=fold_train_x, y=fold_train_y)
            compiled_folds.append((compiled, fold_val_x, fold_val_y))
            logger.info("tuning_compile_fold_complete", fold_idx=fold_idx)

        return compiled_folds

    def _suggest_hyperparams(self, trial: optuna.Trial) -> Hyperparameters:
        regularizers = {
            name: trial.suggest_float(f"reg_{name}", low, high, log=True)
            for name, (low, high) in self.regularizers_range.items()
        }
        graphs = {
            name: trial.suggest_float(f"graph_{name}", low, high, log=True)
            for name, (low, high) in self.graphs_range.items()
        }
        return Hyperparameters(regularizers=regularizers, graphs=graphs)

    def _evaluate_folds(
        self,
        trial: optuna.Trial,
        compiled_folds: list[
            tuple[CompiledPipelineFitProblem[T], pd.DataFrame, pd.Series]
        ],
        hyperparams: Hyperparameters,
    ) -> list[float]:
        fold_scores: list[float] = []
        for fold_idx, (compiled, val_x, val_y) in enumerate(compiled_folds):
            pipeline, _info = self.pipeline_fitter.fit(
                compiled=compiled,
                hyperparameters=hyperparams,
            )
            y_pred = pipeline.transform_and_predict(val_x.copy())
            score = self.scorer(val_y, y_pred)
            fold_scores.append(score)

            # Report intermediate value for pruning
            trial.report(self.aggregator(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned

        return fold_scores

    def _build_initial_params(self, default_value: float) -> dict[str, float]:
        initial_params: dict[str, float] = {}
        for name, (low, high) in self.regularizers_range.items():
            initial_params[f"reg_{name}"] = min(max(default_value, low), high)
        for name, (low, high) in self.graphs_range.items():
            initial_params[f"graph_{name}"] = min(max(default_value, low), high)
        return initial_params

    def _extract_best_hyperparams(self, study: optuna.Study) -> Hyperparameters:
        best_regularizers = {
            name: study.best_params[f"reg_{name}"] for name in self.regularizers_range
        }
        best_graphs = {
            name: study.best_params[f"graph_{name}"] for name in self.graphs_range
        }
        return Hyperparameters(regularizers=best_regularizers, graphs=best_graphs)
