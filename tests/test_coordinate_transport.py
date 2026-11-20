"""Warm starts preserve ordered transform dependencies, not just array shapes."""

import numpy as np
import pytest

from stratified_models import (
    AddIntercept,
    Axis,
    CoordinateError,
    FittedRegressor,
    FixedWidthBins,
    InitialGuess,
    Pipeline,
    RegressionSpec,
    Standardize,
    compose,
    prepare,
)


def test_reordering_normalization_and_binning_cannot_transfer_stratum_coefficients() -> (
    None
):
    x = {"x": np.array([0.5, 1.5])}
    axis = Axis(name="bucket", nodes=(0, 1, 2))
    bins = FixedWidthBins(column="x", output="bucket", origin=-2, width=2, n_bins=3)
    scale = Standardize(columns=("x",))
    model = RegressionSpec(features=("intercept",), axes=(axis,))
    previous = prepare(
        Pipeline(
            preprocessing=compose(scale, bins, AddIntercept(name="intercept")),
            model=model,
        ),
        x,
        np.zeros(2),
    )
    updated = prepare(
        Pipeline(
            preprocessing=compose(bins, scale, AddIntercept(name="intercept")),
            model=model,
        ),
        x,
        np.zeros(2),
    )
    predictor = FittedRegressor(
        pipeline=previous.pipeline,
        transformers=previous.transformers,
        coefficients=np.array([[1.0], [2.0], [3.0]]),
        weights={},
    )
    np.testing.assert_array_equal(predictor.predict(x), [1, 2])
    with pytest.raises(CoordinateError, match="stratum meaning"):
        InitialGuess.from_predictor(predictor).transport_to(updated)


def test_changed_normalization_of_an_unrelated_regressor_can_still_be_transported() -> (
    None
):
    axis = Axis(name="bucket", nodes=(0, 1, 2))
    pipeline = Pipeline(
        preprocessing=compose(
            Standardize(columns=("x",)),
            FixedWidthBins(column="z", output="bucket", origin=0, width=1, n_bins=3),
            AddIntercept(name="intercept"),
        ),
        model=RegressionSpec(features=("x", "intercept"), axes=(axis,)),
    )
    old_x = {"x": np.array([1.0, 3.0]), "z": np.array([0.5, 1.5])}
    new_x = {"x": np.array([10.0, 14.0]), "z": np.array([0.5, 1.5])}
    previous = prepare(pipeline, old_x, np.zeros(2))
    updated = prepare(pipeline, new_x, np.zeros(2))
    predictor = FittedRegressor(
        pipeline=pipeline,
        transformers=previous.transformers,
        coefficients=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        weights={},
    )
    guess = InitialGuess.from_predictor(predictor).transport_to(updated)
    transported = FittedRegressor(
        pipeline=pipeline,
        transformers=updated.transformers,
        coefficients=guess.coefficients,
        weights={},
    )
    np.testing.assert_allclose(transported.predict(old_x), predictor.predict(old_x))
