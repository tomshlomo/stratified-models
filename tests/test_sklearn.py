"""Sklearn adapters preserve clone semantics and independent fitted state."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score

from stratified_models.errors import DataValidationError, SchemaError
from stratified_models.preprocessing import AddIntercept, compose
from stratified_models.sklearn import (
    FitFailedError,
    StratifiedBinaryClassifier,
    StratifiedRegressor,
)
from stratified_models.solvers import NewtonSolver, QuadraticSolver
from stratified_models.specs import L2, BinaryLogisticSpec, Pipeline, RegressionSpec


def test_regressor_clone_is_independent_and_cross_validation_works() -> None:
    recipe = Pipeline(
        preprocessing=compose(AddIntercept(name="intercept")),
        model=RegressionSpec(features=("x0", "intercept")),
    )
    estimator = StratifiedRegressor(recipe, solver=QuadraticSolver())
    copied = clone(estimator)
    assert copied is not estimator
    assert is_regressor(copied)
    x = np.arange(20, dtype=float)[:, None]
    y = 2 * x[:, 0] + 3
    estimator.fit(x, y)
    with pytest.raises(NotFittedError):
        copied.predict(x)
    np.testing.assert_allclose(estimator.predict(x), y, atol=1e-8)
    scores = cross_val_score(copied, x, y, cv=3)
    np.testing.assert_allclose(scores, 1.0, atol=1e-8)


def test_binary_labels_probabilities_and_clone_semantics() -> None:
    recipe = Pipeline(
        preprocessing=compose(AddIntercept(name="intercept")),
        model=BinaryLogisticSpec(
            features=("x0", "intercept"),
            penalties=(L2(name="ridge", weight=0.1, features=("x0",)),),
        ),
    )
    estimator = StratifiedBinaryClassifier(recipe, solver=NewtonSolver())
    assert is_classifier(estimator)
    x = np.linspace(-2, 2, 40)[:, None]
    y = np.where(x[:, 0] > 0, "positive", "negative")
    estimator.fit(x, y)
    np.testing.assert_array_equal(estimator.classes_, ["negative", "positive"])
    np.testing.assert_array_equal(estimator.predict(x), y)
    probabilities = estimator.predict_proba(x)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert estimator.score(x, y) == 1.0
    copied = clone(estimator)
    assert not hasattr(copied, "predictor_")
    assert not hasattr(copied, "classes_")
    with pytest.raises(DataValidationError, match="exactly two"):
        estimator.fit(x, np.zeros(len(x)))
    with pytest.raises(NotFittedError):
        estimator.predict(x)


def test_numerical_failure_is_an_explicit_adapter_exception() -> None:
    recipe = Pipeline(
        preprocessing=compose(AddIntercept(name="intercept")),
        model=BinaryLogisticSpec(
            features=("x0", "intercept"),
            penalties=(L2(name="ridge", weight=0.1, features=("x0",)),),
        ),
    )
    estimator = StratifiedBinaryClassifier(
        recipe, solver=NewtonSolver(max_iterations=1, atol=1e-12, rtol=1e-12)
    )
    x = np.linspace(-2, 2, 40)[:, None]
    y = (x[:, 0] > 0).astype(int)
    with pytest.raises(FitFailedError) as failure:
        estimator.fit(x, y)
    assert failure.value.result.diagnostics.converged is False
    with pytest.raises(NotFittedError):
        estimator.predict(x)


def test_classifier_preserves_pandas_alignment_checks() -> None:
    pandas = pytest.importorskip("pandas")
    recipe = Pipeline(
        preprocessing=compose(AddIntercept(name="intercept")),
        model=BinaryLogisticSpec(
            features=("x0", "intercept"),
            penalties=(L2(name="ridge", weight=0.1, features=("x0",)),),
        ),
    )
    estimator = StratifiedBinaryClassifier(recipe, solver=NewtonSolver())
    x = pandas.DataFrame({"x0": [-2.0, -1.0, 1.0, 2.0]}, index=[4, 5, 6, 7])
    estimator.fit(x, ["negative", "negative", "positive", "positive"])
    np.testing.assert_array_equal(
        estimator.predict(x), ["negative", "negative", "positive", "positive"]
    )
    misaligned = pandas.Series([0, 0, 1, 1], index=[7, 6, 5, 4])
    with pytest.raises(SchemaError, match="indexes"):
        estimator.fit(x, misaligned)
