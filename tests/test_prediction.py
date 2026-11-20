"""Prediction rejects arithmetic failure rather than returning invalid outputs."""

from collections.abc import Generator

import numpy as np
import pytest

from stratified_models import (
    BinaryLogisticSpec,
    DataValidationError,
    FeatureBatch,
    FittedBinaryClassifier,
    FittedRegressor,
    FittedTransformChain,
    Pipeline,
    RegressionSpec,
    SchemaError,
    compose,
)


def test_finite_regression_inputs_with_overflow_raise_explicitly() -> None:
    predictor = FittedRegressor(
        pipeline=Pipeline(
            preprocessing=compose(), model=RegressionSpec(features=("x",))
        ),
        transformers=FittedTransformChain(),
        coefficients=np.array([[1e308]]),
        weights={},
    )
    with pytest.raises(DataValidationError, match="prediction arithmetic"):
        predictor.predict({"x": np.array([2.0])})


def test_logistic_cancellation_does_not_silently_produce_nan_probabilities() -> None:
    predictor = FittedBinaryClassifier(
        pipeline=Pipeline(
            preprocessing=compose(), model=BinaryLogisticSpec(features=("a", "b"))
        ),
        transformers=FittedTransformChain(),
        coefficients=np.array([[1e308, -1e308]]),
        weights={},
    )
    with pytest.raises(DataValidationError, match="prediction arithmetic"):
        predictor.predict_proba({"a": np.array([2.0]), "b": np.array([2.0])})


def test_empty_classification_labels_keep_their_integer_type() -> None:
    predictor = FittedBinaryClassifier(
        pipeline=Pipeline(
            preprocessing=compose(), model=BinaryLogisticSpec(features=("x",))
        ),
        transformers=FittedTransformChain(),
        coefficients=np.array([[1.0]]),
        weights={},
    )
    assert predictor.predict({"x": np.empty(0)}).dtype == np.dtype(np.int64)
    assert predictor.predict_proba({"x": np.empty(0)}).shape == (0, 2)


def test_prediction_source_must_honor_its_declared_column_schema() -> None:
    class BadSource:
        snapshot_id = "unchanged"
        n_rows = 1
        column_names = ("x",)

        def batches(self) -> Generator[FeatureBatch]:
            yield FeatureBatch(
                columns={"x": np.ones(1), "unexpected": np.ones(1)},
                row_ids=np.array([0]),
            )

    predictor = FittedRegressor(
        pipeline=Pipeline(
            preprocessing=compose(), model=RegressionSpec(features=("x",))
        ),
        transformers=FittedTransformChain(),
        coefficients=np.ones((1, 1)),
        weights={},
    )
    with (
        predictor.predict_batches(BadSource()) as stream,
        pytest.raises(SchemaError, match="schema"),
    ):
        next(stream)
