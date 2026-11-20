import importlib
import subprocess
import sys
from collections.abc import Generator

import numpy as np
import pytest

from stratified_models.data import TrainingBatch, source_from_data
from stratified_models.errors import (
    DataValidationError,
    ReplayError,
    SchemaError,
    ValidationError,
)
from stratified_models.fit import FitSuccess, fit_batches
from stratified_models.prepared import PreparationOptions
from stratified_models.preprocessing import AddIntercept, Standardize, TransformChain
from stratified_models.solvers import ADMM, AutoSolver, DirectSolve, SolverCache
from stratified_models.specs import (
    L2,
    BinaryLogisticSpec,
    Pipeline,
    RegressionSpec,
    WeightRef,
)
from stratified_models.tuning import (
    R2,
    BinaryLogLoss,
    Choice,
    Fold,
    LogUniform,
    Maximize,
    Minimize,
    SquaredError,
    TrialBudget,
    TuningFailure,
    TuningSuccess,
    Uniform,
    WeightSpace,
    tune,
)


def _recipe() -> Pipeline[RegressionSpec]:
    return Pipeline(
        preprocessing=TransformChain(
            steps=(Standardize(columns=("x",)), AddIntercept(name="intercept"))
        ),
        model=RegressionSpec(
            features=("x", "intercept"),
            penalties=(L2(name="ridge", weight=0.7, features=("x",)),),
        ),
    )


def _folds() -> tuple[Fold, ...]:
    return (
        Fold(
            train=source_from_data(
                {"x": [-3.0, -1.0, 0.0, 2.0]}, [-5.0, -1.0, 1.0, 5.0], batch_size=3
            ),
            validation=source_from_data({"x": [1.0, 4.0]}, [3.0, 9.0], batch_size=1),
        ),
        Fold(
            train=source_from_data(
                {"x": [10.0, 11.0, 13.0, 16.0]}, [21.0, 23.0, 27.0, 33.0], batch_size=2
            ),
            validation=source_from_data(
                {"x": [12.0, 15.0, 20.0]}, [25.0, 31.0, 41.0], batch_size=2
            ),
        ),
    )


def test_metrics_are_invariant_to_uneven_partitions() -> None:
    targets = np.array([0.0, 4.0, 2.0, 5.0, 3.0])
    predictions = np.array([1.0, 2.0, 2.5, 4.0, 3.1])
    for metric in (SquaredError(), R2()):
        all_at_once = metric.finalize(
            metric.update(metric.initial(), targets, predictions)
        )
        state = metric.initial()
        for start, stop in ((0, 1), (1, 3), (3, 5)):
            state = metric.update(state, targets[start:stop], predictions[start:stop])
        assert metric.finalize(state) == pytest.approx(all_at_once)
    assert SquaredError().finalize(
        SquaredError().update(SquaredError().initial(), targets, predictions)
    ) == pytest.approx(np.mean((targets - predictions) ** 2))
    for metric in (SquaredError(), BinaryLogLoss()):
        with pytest.raises(DataValidationError, match="empty"):
            metric.finalize(metric.initial())


def test_binary_log_loss_handles_extreme_probabilities_and_validates_inputs() -> None:
    metric = BinaryLogLoss()
    targets = np.array([0.0, 1.0, 1.0])
    positive = np.array([0.0, 1.0, 0.2])
    state = metric.update(
        metric.initial(), targets, np.column_stack((1 - positive, positive))
    )
    assert metric.finalize(state) == pytest.approx(-np.log(0.2) / 3, abs=1e-14)
    with pytest.raises(DataValidationError, match="zero/one"):
        metric.update(metric.initial(), np.array([2.0]), np.array([0.5]))
    with pytest.raises(DataValidationError, match="probabilities"):
        metric.update(metric.initial(), np.array([1.0]), np.array([1.1]))


def test_search_configuration_is_owned_and_duplicate_or_invalid_ranges_fail() -> None:
    choices = [0.0, 0.1]
    ranges = {"ridge": Choice(values=choices)}
    space = WeightSpace(ranges=ranges)
    choices.append(10.0)
    ranges.clear()
    assert space.ranges["ridge"].values == (0.0, 0.1)
    with pytest.raises(TypeError):
        space.ranges["other"] = Uniform(low=0, high=1)
    with pytest.raises(ValidationError, match="duplicate"):
        WeightSpace(
            ranges={
                "ridge": Uniform(low=0, high=1),
                WeightRef(name="ridge"): Uniform(low=0, high=2),
            }
        )
    for low, high in ((-1, 1), (2, 1), (0, np.inf)):
        with pytest.raises(ValidationError):
            Uniform(low=low, high=high)
    with pytest.raises(ValidationError):
        LogUniform(low=0, high=1)


def test_unknown_search_names_are_rejected_before_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("stratified_models.tuning")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid search must not prepare or scan a fold")

    monkeypatch.setattr(module, "prepare_batches", forbidden)
    with pytest.raises(ValidationError, match="unknown penalties"):
        tune(
            _recipe(),
            folds=_folds(),
            search=WeightSpace(ranges={"typo": Uniform(low=0, high=1)}),
            metric=Minimize(metric=SquaredError()),
            solver=DirectSolve(),
            budget=TrialBudget(trials=1),
        )


def test_tuning_matches_independent_fits_and_pools_validation_rows() -> None:
    pytest.importorskip("optuna")
    recipe = _recipe()
    result = tune(
        recipe,
        folds=_folds(),
        search=WeightSpace(ranges={recipe.weight("ridge"): Choice(values=(0.3,))}),
        metric=Minimize(metric=SquaredError()),
        solver=DirectSolve(),
        budget=TrialBudget(trials=1),
    )
    assert isinstance(result, TuningSuccess)
    squared_error = 0.0
    count = 0
    for fold, report in zip(_folds(), result.trials[0].folds, strict=True):
        fitted = fit_batches(result.best_recipe, fold.train, solver=DirectSolve())
        assert isinstance(fitted, FitSuccess)
        fold_sum = 0.0
        fold_count = 0
        for batch in fold.validation.batches():
            residual = fitted.predictor.predict_batch(batch) - batch.targets
            fold_sum += residual @ residual
            fold_count += len(residual)
        assert report.score == pytest.approx(fold_sum / fold_count)
        squared_error += fold_sum
        count += fold_count
    assert result.best_score == pytest.approx(squared_error / count)
    assert recipe.model.penalties[0].weight == 0.7
    assert result.best_recipe.model.penalties[0].weight == 0.3
    with pytest.raises(TypeError):
        result.weights["ridge"] = 1.0


def test_folds_have_independent_transforms_sessions_and_shared_bounded_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("optuna")
    fitting = importlib.import_module("stratified_models.fit")
    original = fitting.SolveSession
    sessions = []

    class ObservedSession(original):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.calls = 0
            sessions.append(self)

        def solve(self, *args: object, **kwargs: object) -> object:
            self.calls += 1
            return super().solve(*args, **kwargs)

    monkeypatch.setattr(fitting, "SolveSession", ObservedSession)
    cache = SolverCache(max_bytes=4_096)
    result = tune(
        _recipe(),
        folds=_folds(),
        search=WeightSpace(ranges={"ridge": Choice(values=(0.3,))}),
        metric=Minimize(metric=SquaredError()),
        solver=DirectSolve(),
        budget=TrialBudget(trials=3),
        cache=cache,
        options=PreparationOptions(statistics_max_bytes=168),
    )
    assert isinstance(result, TuningSuccess)
    assert len(sessions) == 2
    assert [session.calls for session in sessions] == [3, 3]
    assert all(session.cache is cache for session in sessions)
    assert all(session.prepared.statistics is None for session in sessions)
    means = [session.prepared.transformers.steps[0].means[0] for session in sessions]
    assert means == [-0.5, 12.5]
    assert cache.snapshot().bytes_used <= cache.max_bytes


def test_seed_replays_search_and_direction_is_explicit() -> None:
    pytest.importorskip("optuna")
    arguments = {
        "folds": _folds(),
        "search": WeightSpace(ranges={"ridge": Uniform(low=0.01, high=1.0)}),
        "metric": Maximize(metric=R2()),
        "solver": DirectSolve(),
        "budget": TrialBudget(trials=3),
        "seed": 17,
    }
    first = tune(_recipe(), **arguments)
    second = tune(_recipe(), **arguments)
    assert isinstance(first, TuningSuccess)
    assert isinstance(second, TuningSuccess)
    assert first.weights == second.weights
    assert first.best_score == second.best_score
    assert [trial.weights for trial in first.trials] == [
        trial.weights for trial in second.trials
    ]
    assert first.best_score == max(trial.score for trial in first.trials)


def test_all_nonconverged_trials_return_typed_failure() -> None:
    pytest.importorskip("optuna")
    result = tune(
        _recipe(),
        folds=_folds(),
        search=WeightSpace(ranges={"ridge": Choice(values=(0.3,))}),
        metric=Minimize(metric=SquaredError()),
        solver=ADMM(max_iterations=1, atol=1e-14, rtol=1e-14),
        budget=TrialBudget(trials=2),
    )
    assert isinstance(result, TuningFailure)
    assert result.reason.code == "no_successful_trials"
    assert len(result.trials) == 2
    assert all(
        trial.status == "failed" and trial.score is None for trial in result.trials
    )
    assert all(trial.folds[-1].failure is not None for trial in result.trials)


def test_binary_tuning_streams_probabilities_into_log_loss() -> None:
    pytest.importorskip("optuna")
    recipe = Pipeline(
        preprocessing=TransformChain(steps=(AddIntercept(name="intercept"),)),
        model=BinaryLogisticSpec(
            features=("x", "intercept"),
            penalties=(L2(name="ridge", weight=0.5, features=("x",)),),
        ),
    )
    training = source_from_data(
        {"x": [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]}, [0, 1, 0, 1, 0, 1], batch_size=2
    )
    validation = source_from_data({"x": [-1.5, 0.0, 1.5]}, [0, 1, 1], batch_size=2)
    result = tune(
        recipe,
        folds=(Fold(train=training, validation=validation),),
        search=WeightSpace(ranges={"ridge": Choice(values=(0.1,))}),
        metric=Minimize(metric=BinaryLogLoss()),
        solver=AutoSolver(),
        budget=TrialBudget(trials=1),
    )
    assert isinstance(result, TuningSuccess)
    assert 0 < result.best_score < 2
    independently_fit = fit_batches(result.best_recipe, training, solver=AutoSolver())
    assert isinstance(independently_fit, FitSuccess)
    metric = BinaryLogLoss()
    state = metric.initial()
    for batch in validation.batches():
        state = metric.update(
            state, batch.targets, independently_fit.predictor.predict_proba_batch(batch)
        )
    assert result.best_score == pytest.approx(metric.finalize(state))


def test_validation_replay_changes_raise_instead_of_becoming_failed_trials() -> None:
    pytest.importorskip("optuna")
    stable = _folds()[0]

    class ChangingValidation:
        snapshot_id = "claims-stability"
        n_rows = 2
        column_names = ("x",)

        def __init__(self) -> None:
            self.passes = 0

        def batches(self) -> Generator[TrainingBatch]:
            self.passes += 1
            yield TrainingBatch(
                columns={"x": [self.passes, 3.0]}, targets=[0.0, 1.0], row_ids=[0, 1]
            )

    with pytest.raises(ReplayError, match="changed between"):
        tune(
            _recipe(),
            folds=(Fold(train=stable.train, validation=ChangingValidation()),),
            search=WeightSpace(ranges={"ridge": Choice(values=(0.3,))}),
            metric=Minimize(metric=SquaredError()),
            solver=DirectSolve(),
            budget=TrialBudget(trials=2),
        )


def test_probability_metric_rejects_regression_before_scanning() -> None:
    with pytest.raises(SchemaError, match="binary"):
        tune(
            _recipe(),
            folds=_folds(),
            search=WeightSpace(ranges={}),
            metric=Minimize(metric=BinaryLogLoss()),
            solver=DirectSolve(),
            budget=TrialBudget(trials=1),
        )


def test_importing_tuning_does_not_import_optuna() -> None:
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stratified_models.tuning; assert 'optuna' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
