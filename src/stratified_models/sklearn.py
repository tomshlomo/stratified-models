"""Optional mutable scikit-learn adapters around immutable native predictors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Self

import numpy as np
from numpy.typing import NDArray

from stratified_models.errors import (
    DataValidationError,
    OptionalDependencyError,
    SchemaError,
    StratifiedModelsError,
)

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.utils.multiclass import check_classification_targets, unique_labels
    from sklearn.utils.validation import check_is_fitted
except ImportError as error:
    msg = "The sklearn adapter requires the 'sklearn' optional integration."
    raise OptionalDependencyError(msg) from error

if TYPE_CHECKING:
    from stratified_models.fit import FitFailure
    from stratified_models.predict import FittedBinaryClassifier, FittedRegressor
    from stratified_models.prepared import PreparationOptions
    from stratified_models.solvers import Solver
    from stratified_models.solvers.types import Diagnostics
    from stratified_models.specs import BinaryLogisticSpec, Pipeline, RegressionSpec


class FitFailedError(StratifiedModelsError, RuntimeError):
    """A native numerical solve failed; inspect ``result`` for diagnostics."""

    def __init__(self, result: FitFailure) -> None:
        self.result = result
        super().__init__(f"Stratified model fitting failed: {result.reason}")


def _input_columns(X: object) -> tuple[str, ...] | None:
    names = tuple(X) if isinstance(X, Mapping) else getattr(X, "columns", None)
    if names is None:
        return None
    validated: list[str] = []
    for name in names:
        if not isinstance(name, str):
            return None
        validated.append(name)
    return tuple(validated)


def _metadata(
    estimator: StratifiedRegressor | StratifiedBinaryClassifier, X: object
) -> None:
    names = _input_columns(X)
    if names is not None:
        estimator.n_features_in_ = len(names)
        estimator.feature_names_in_ = np.asarray(names, dtype=object)
        return
    values = np.asarray(X)
    if values.ndim != 2:
        msg = "Features must be a two-dimensional array or named column container."
        raise SchemaError(msg)
    estimator.n_features_in_ = values.shape[1]


def _clear_fitted(estimator: BaseEstimator) -> None:
    for name in (
        "predictor_",
        "diagnostics_",
        "n_features_in_",
        "feature_names_in_",
        "classes_",
        "coef_",
    ):
        if hasattr(estimator, name):
            delattr(estimator, name)


def _check_input(
    estimator: StratifiedRegressor | StratifiedBinaryClassifier, X: object
) -> None:
    check_is_fitted(estimator, "predictor_")
    names = _input_columns(X)
    expected_names = getattr(estimator, "feature_names_in_", None)
    if (
        expected_names is not None
        and names is not None
        and tuple(expected_names) != names
    ):
        msg = "Prediction columns must match fitting columns and their order."
        raise SchemaError(msg)
    count = (
        len(names)
        if names is not None
        else np.asarray(X).shape[1]
        if np.asarray(X).ndim == 2
        else None
    )
    if count != estimator.n_features_in_:
        msg = f"Expected {estimator.n_features_in_} input columns, received {count}."
        raise SchemaError(msg)


def _check_target_alignment(X: object, y: object) -> None:
    x_index = getattr(X, "index", None)
    y_index = getattr(y, "index", None)
    # Python lists expose an unrelated ``index`` method.
    if x_index is not None and hasattr(x_index, "is_unique") and not x_index.is_unique:
        msg = "Feature indexes must be unique."
        raise SchemaError(msg)
    if y_index is not None and hasattr(y_index, "is_unique") and not y_index.is_unique:
        msg = "Target indexes must be unique."
        raise SchemaError(msg)
    if (
        x_index is not None
        and y_index is not None
        and hasattr(x_index, "equals")
        and hasattr(y_index, "is_unique")
        and not x_index.equals(y_index)
    ):
        msg = "Feature and target indexes must match exactly and in order."
        raise SchemaError(msg)


class StratifiedRegressor(RegressorMixin, BaseEstimator):
    """A cloneable sklearn regressor with an explicit native solver."""

    n_features_in_: int
    feature_names_in_: NDArray[np.object_]
    predictor_: FittedRegressor
    diagnostics_: Diagnostics
    coef_: NDArray[np.float64]

    def __init__(
        self,
        recipe: Pipeline[RegressionSpec],
        *,
        solver: Solver,
        options: PreparationOptions | None = None,
    ) -> None:
        self.recipe = recipe
        self.solver = solver
        self.options = options

    def fit(self, X: object, y: object) -> Self:
        from stratified_models.fit import FitFailure, fit

        _clear_fitted(self)
        _metadata(self, X)
        result = fit(self.recipe, X, y, solver=self.solver, options=self.options)
        if isinstance(result, FitFailure):
            _clear_fitted(self)
            raise FitFailedError(result)
        self.predictor_ = result.predictor
        self.diagnostics_ = result.diagnostics
        self.coef_ = result.predictor.coefficients
        return self

    def predict(self, X: object) -> NDArray[np.float64]:
        _check_input(self, X)
        return self.predictor_.predict(X)


class StratifiedBinaryClassifier(ClassifierMixin, BaseEstimator):
    """Map arbitrary sklearn binary labels onto the native zero/one task."""

    n_features_in_: int
    feature_names_in_: NDArray[np.object_]
    predictor_: FittedBinaryClassifier
    diagnostics_: Diagnostics
    coef_: NDArray[np.float64]
    classes_: NDArray

    def __init__(
        self,
        recipe: Pipeline[BinaryLogisticSpec],
        *,
        solver: Solver,
        options: PreparationOptions | None = None,
    ) -> None:
        self.recipe = recipe
        self.solver = solver
        self.options = options

    def fit(self, X: object, y: object) -> Self:
        from stratified_models.fit import FitFailure, fit

        _clear_fitted(self)
        _metadata(self, X)
        _check_target_alignment(X, y)
        targets = np.asarray(y)
        if targets.ndim != 1:
            msg = "Binary classifier targets must be one-dimensional."
            raise DataValidationError(msg)
        check_classification_targets(targets)
        classes = unique_labels(targets)
        if len(classes) != 2:
            msg = "The binary classifier requires exactly two observed classes."
            raise DataValidationError(msg)
        encoded = (targets == classes[1]).astype(np.float64)
        result = fit(self.recipe, X, encoded, solver=self.solver, options=self.options)
        if isinstance(result, FitFailure):
            _clear_fitted(self)
            raise FitFailedError(result)
        self.predictor_ = result.predictor
        self.diagnostics_ = result.diagnostics
        self.coef_ = result.predictor.coefficients
        self.classes_ = np.array(classes, copy=True)
        return self

    def predict(self, X: object) -> NDArray:
        _check_input(self, X)
        codes = self.predictor_.predict(X)
        return self.classes_[codes]

    def predict_proba(self, X: object) -> NDArray[np.float64]:
        _check_input(self, X)
        return self.predictor_.predict_proba(X)

    def decision_function(self, X: object) -> NDArray[np.float64]:
        _check_input(self, X)
        return self.predictor_.decision_function(X)


StratifiedClassifier = StratifiedBinaryClassifier
