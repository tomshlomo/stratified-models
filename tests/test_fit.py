"""Preparation/solver boundary checks using the actual public data contracts."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from stratified_models.data import source_from_data
from stratified_models.errors import CoordinateError
from stratified_models.fit import (
    FitFailure,
    FitSuccess,
    InitialGuess,
    SolveSession,
    fit,
    fit_batches,
)
from stratified_models.predict import FittedBinaryClassifier, FittedRegressor
from stratified_models.prepared import PreparationOptions, prepare, prepare_batches
from stratified_models.preprocessing import AddIntercept, Standardize, compose
from stratified_models.solvers import (
    ADMM,
    ConjugateGradient,
    DirectSolve,
    NewtonSolver,
    NumericalSuccess,
    SolverCache,
    solve,
)
from stratified_models.specs import (
    L1,
    L2,
    Axis,
    BinaryLogisticSpec,
    Graph,
    Laplacian,
    Pipeline,
    RegressionSpec,
)


def _recipe(task: Literal["regression", "binary"], l1: float = 0.0) -> Pipeline:
    axis = Axis(name="region", nodes=("west", "east"))
    spec_type = RegressionSpec if task == "regression" else BinaryLogisticSpec
    return Pipeline(
        preprocessing=compose(
            Standardize(columns=("x",)), AddIntercept(name="intercept")
        ),
        model=spec_type(
            features=("intercept", "x"),
            axes=(axis,),
            penalties=(
                L1(name="sparse", weight=l1, features=("x",)),
                L2(name="ridge", weight=0.1, features=("x",)),
                Laplacian(
                    name="space",
                    weight=0.05,
                    features=("intercept", "x"),
                    graph=Graph.path(axis=axis),
                ),
            ),
        ),
    )


def _data(
    task: Literal["regression", "binary"],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(31)
    x = rng.normal(size=40)
    region = np.where(np.arange(40) % 2, "west", "east")
    prediction = 0.4 + 1.3 * x + 0.5 * (region == "west")
    target = (
        prediction
        if task == "regression"
        else rng.binomial(1, 1 / (1 + np.exp(-prediction))).astype(float)
    )
    return {"x": x, "region": region}, target


@pytest.mark.parametrize("task", ["regression", "binary"])
@pytest.mark.parametrize("use_admm", [False, True])
def test_true_streaming_and_unequal_batches_match_inmemory_preparation(
    task: Literal["regression", "binary"], use_admm: bool
) -> None:
    columns, target = _data(task)
    recipe = _recipe(task, l1=0.02 if use_admm else 0.0)
    settings = PreparationOptions(statistics_max_bytes=0)
    in_memory = prepare(recipe, columns, target, options=settings)
    streamed = prepare_batches(
        recipe, source_from_data(columns, target, batch_size=7), options=settings
    )
    assert in_memory.statistics is None and streamed.statistics is None
    strategy = (
        ADMM(atol=1e-6, rtol=1e-5)
        if use_admm
        else ConjugateGradient()
        if task == "regression"
        else NewtonSolver()
    )
    first = solve(in_memory, solver=strategy, weights={}, cache=SolverCache())
    second = solve(streamed, solver=strategy, weights={}, cache=SolverCache())
    assert isinstance(first, NumericalSuccess), first
    assert isinstance(second, NumericalSuccess), second
    assert np.allclose(first.coefficients, second.coefficients, atol=1e-6)
    assert first.diagnostics.objective.total == pytest.approx(
        second.diagnostics.objective.total, abs=1e-8
    )
    assert first.diagnostics.rows_scanned == first.diagnostics.source_passes * 40
    assert second.diagnostics.rows_scanned == second.diagnostics.source_passes * 40


def test_fold_preparation_fits_independent_standardization_without_mutating_inputs() -> (
    None
):
    columns, target = _data("regression")
    original = columns["x"].copy()
    train = np.arange(20)
    training_columns = {name: values[train] for name, values in columns.items()}
    fold = prepare(_recipe("regression"), training_columns, target[train])
    transformed = np.concatenate([batch.x[:, 1] for batch in fold.iter_numeric()])
    assert np.mean(transformed) == pytest.approx(0.0, abs=1e-14)
    assert np.var(transformed) == pytest.approx(1.0)
    assert np.array_equal(columns["x"], original)
    assert fold.n_samples == 20


def test_functional_fit_returns_task_specific_predictors_and_typed_failures() -> None:
    regression_columns, regression_target = _data("regression")
    regression = fit(
        _recipe("regression"),
        regression_columns,
        regression_target,
        solver=DirectSolve(),
    )
    assert isinstance(regression, FitSuccess)
    assert isinstance(regression.predictor, FittedRegressor)
    assert regression.predictor.predict(regression_columns).shape == (40,)
    binary_columns, binary_target = _data("binary")
    classification = fit_batches(
        _recipe("binary"),
        source_from_data(binary_columns, binary_target, batch_size=11),
        solver=NewtonSolver(),
    )
    assert isinstance(classification, FitSuccess)
    assert isinstance(classification.predictor, FittedBinaryClassifier)
    probability = classification.predictor.predict_proba(binary_columns)
    assert probability.shape == (40, 2)
    assert np.allclose(probability.sum(axis=1), 1)
    assert set(classification.predictor.predict(binary_columns)) <= {0, 1}
    failed = fit(
        _recipe("binary"),
        binary_columns,
        binary_target,
        solver=NewtonSolver(max_passes=1),
    )
    assert isinstance(failed, FitFailure)
    assert failed.reason.code == "pass_limit"


def test_sessions_share_factors_but_changed_targets_and_designs_are_correctly_isolated() -> (
    None
):
    columns, target = _data("regression")
    pipeline = _recipe("regression")
    prepared = prepare(pipeline, columns, target)
    cache = SolverCache()
    first = SolveSession(prepared, solver=DirectSolve(), cache=cache).solve()
    second = SolveSession(prepared, solver=DirectSolve(), cache=cache).solve()
    shifted_target = SolveSession(
        prepare(pipeline, columns, 2 * target), solver=DirectSolve(), cache=cache
    ).solve()
    changed_columns = {**columns, "x": columns["x"] ** 3}
    changed_design = SolveSession(
        prepare(pipeline, changed_columns, target), solver=DirectSolve(), cache=cache
    ).solve()
    assert isinstance(first, FitSuccess) and isinstance(second, FitSuccess)
    assert isinstance(shifted_target, FitSuccess) and isinstance(
        changed_design, FitSuccess
    )
    assert second.diagnostics.cache.hits == 1
    assert shifted_target.diagnostics.cache.hits == 1
    assert shifted_target.diagnostics.cache.factorizations == 0
    assert np.allclose(
        shifted_target.predictor.coefficients, 2 * first.predictor.coefficients
    )
    assert changed_design.diagnostics.cache.factorizations == 1
    different_weight = SolveSession(prepared, solver=DirectSolve(), cache=cache).solve(
        weight_overrides={pipeline.weight("ridge"): 0.7}
    )
    assert isinstance(different_weight, FitSuccess)
    assert different_weight.diagnostics.cache.factorizations == 1
    assert first.predictor.weights["ridge"] == 0.1


def test_explicit_affine_warm_start_transport_preserves_prediction_meaning() -> None:
    columns, target = _data("regression")
    pipeline = _recipe("regression")
    original = fit(pipeline, columns, target, solver=DirectSolve())
    assert isinstance(original, FitSuccess)
    guess = InitialGuess.from_predictor(original.predictor)
    changed = {**columns, "x": 2 * columns["x"] + 4}
    prepared = prepare(pipeline, changed, target)
    session = SolveSession(prepared, solver=DirectSolve())
    with pytest.raises(CoordinateError, match="transport"):
        session.solve(initial=guess)
    transported = guess.transport_to(prepared)
    reexpressed = FittedRegressor(
        pipeline=prepared.pipeline,
        transformers=prepared.transformers,
        coefficients=transported.coefficients,
        weights=original.predictor.weights,
    )
    assert np.allclose(
        original.predictor.predict(columns), reexpressed.predict(columns), atol=1e-12
    )
    warm = session.solve(initial=transported)
    cold = SolveSession(prepared, solver=DirectSolve()).solve()
    assert isinstance(warm, FitSuccess) and isinstance(cold, FitSuccess)
    assert np.allclose(warm.predictor.coefficients, cold.predictor.coefficients)


def test_admm_session_reuses_dual_continuation_and_preserves_previous_outputs() -> None:
    columns, target = _data("regression")
    prepared = prepare(_recipe("regression", l1=0.04), columns, target)
    session = SolveSession(prepared, solver=ADMM())
    first = session.solve()
    assert isinstance(first, FitSuccess)
    previous = first.predictor.coefficients.copy()
    second = session.solve()
    assert isinstance(second, FitSuccess)
    assert second.diagnostics.iterations <= 2
    assert np.array_equal(first.predictor.coefficients, previous)
