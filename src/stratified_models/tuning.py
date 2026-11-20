"""Optional, fold-isolated tuning with bounded streaming metric state.

Only this module's ``tune`` operation imports Optuna. A tuning result contains
immutable reports and a new recipe; refitting the selected recipe is explicit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol

import attrs
import numpy as np
from numpy.typing import NDArray

from stratified_models.data import TrainingBatch, TrainingSource
from stratified_models.errors import (
    DataValidationError,
    OptionalDependencyError,
    ReplayError,
    SchemaError,
    ValidationError,
)
from stratified_models.prepared import PreparationOptions, prepare_batches
from stratified_models.solvers import Solver, SolverCache
from stratified_models.solvers.types import FailureReason
from stratified_models.specs import ModelSpec, Pipeline, WeightRef, validate_name

if TYPE_CHECKING:
    import optuna

    from stratified_models.fit import SolveSession
    from stratified_models.predict import FittedBinaryClassifier, FittedRegressor

type FloatArray = NDArray[np.float64]


def _tuple[T](values: Iterable[T]) -> tuple[T, ...]:
    return tuple(values)


def _range(low: float, high: float, *, logarithmic: bool) -> None:
    if not math.isfinite(low) or not math.isfinite(high) or low < 0 or high < low:
        message = "weight ranges require finite nonnegative low <= high"
        raise ValidationError(message)
    if logarithmic and low == 0:
        message = "logarithmic weight ranges require low > 0"
        raise ValidationError(message)


@attrs.frozen(kw_only=True)
class LogUniform:
    low: float = attrs.field(converter=float)
    high: float = attrs.field(converter=float)

    def __attrs_post_init__(self) -> None:
        _range(self.low, self.high, logarithmic=True)


@attrs.frozen(kw_only=True)
class Uniform:
    low: float = attrs.field(converter=float)
    high: float = attrs.field(converter=float)

    def __attrs_post_init__(self) -> None:
        _range(self.low, self.high, logarithmic=False)


def _choices(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


@attrs.frozen(kw_only=True)
class Choice:
    values: tuple[float, ...] = attrs.field(converter=_choices)

    def __attrs_post_init__(self) -> None:
        if not self.values or any(
            not math.isfinite(value) or value < 0 for value in self.values
        ):
            message = "weight choices must be nonempty, finite and nonnegative"
            raise ValidationError(message)
        if len(set(self.values)) != len(self.values):
            message = "weight choices must be unique"
            raise ValidationError(message)


type Distribution = LogUniform | Uniform | Choice
type RangeInput = (
    Mapping[str, Distribution]
    | Mapping[WeightRef, Distribution]
    | Mapping[str | WeightRef, Distribution]
)


def _ranges(
    values: RangeInput,
) -> Mapping[str, Distribution]:
    ranges: dict[str, Distribution] = {}
    for key, distribution in values.items():
        name = key.name if isinstance(key, WeightRef) else key
        validate_name(name, kind="weight search name")
        if name in ranges:
            message = f"duplicate weight search name {name!r}"
            raise ValidationError(message)
        if not isinstance(distribution, (LogUniform, Uniform, Choice)):
            message = f"unsupported distribution for penalty {name!r}"
            raise ValidationError(message)
        ranges[name] = distribution
    return MappingProxyType(ranges)


@attrs.frozen(kw_only=True, init=False)
class WeightSpace:
    ranges: Mapping[str, Distribution]

    def __init__(self, *, ranges: RangeInput) -> None:
        object.__setattr__(self, "ranges", _ranges(ranges))


class StreamingMetric[S](Protocol):
    @property
    def prediction_kind(self) -> Literal["prediction", "probability"]: ...

    def initial(self) -> S: ...

    def update(self, state: S, targets: FloatArray, predictions: FloatArray) -> S: ...

    def finalize(self, state: S) -> float: ...


@attrs.frozen(kw_only=True)
class _ErrorState:
    count: int = 0
    total: float = 0.0


def _metric_arrays(
    targets: FloatArray, predictions: FloatArray
) -> tuple[FloatArray, FloatArray]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 1 or predictions.shape != targets.shape:
        message = "streaming metric targets and predictions must be aligned vectors"
        raise SchemaError(message)
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(predictions)):
        message = "streaming metrics require finite targets and predictions"
        raise DataValidationError(message)
    return targets, predictions


@attrs.frozen(kw_only=True)
class SquaredError:
    """Mean squared prediction error, pooled over all validation rows."""

    @property
    def prediction_kind(self) -> Literal["prediction"]:
        return "prediction"

    def initial(self) -> _ErrorState:
        return _ErrorState()

    def update(
        self, state: _ErrorState, targets: FloatArray, predictions: FloatArray
    ) -> _ErrorState:
        targets, predictions = _metric_arrays(targets, predictions)
        residuals = predictions - targets
        return _ErrorState(
            count=state.count + len(targets),
            total=state.total + float(residuals @ residuals),
        )

    def finalize(self, state: _ErrorState) -> float:
        if state.count == 0:
            message = "cannot score an empty validation source"
            raise DataValidationError(message)
        return state.total / state.count


@attrs.frozen(kw_only=True)
class BinaryLogLoss:
    """Mean binary log loss; exact zero/one probabilities are clipped by eps."""

    eps: float = 1e-15

    def __attrs_post_init__(self) -> None:
        if not math.isfinite(self.eps) or not 0 < self.eps < 0.5:
            message = "log-loss eps must be finite and between zero and one half"
            raise ValidationError(message)

    @property
    def prediction_kind(self) -> Literal["probability"]:
        return "probability"

    def initial(self) -> _ErrorState:
        return _ErrorState()

    def update(
        self, state: _ErrorState, targets: FloatArray, predictions: FloatArray
    ) -> _ErrorState:
        predictions = np.asarray(predictions, dtype=np.float64)
        if predictions.ndim == 2 and predictions.shape[1] == 2:
            if not np.allclose(predictions.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
                message = "binary probability rows must sum to one"
                raise DataValidationError(message)
            predictions = predictions[:, 1]
        targets, predictions = _metric_arrays(targets, predictions)
        if not np.all((targets == 0) | (targets == 1)):
            message = "binary log loss requires zero/one targets"
            raise DataValidationError(message)
        if np.any(predictions < 0) or np.any(predictions > 1):
            message = "binary log loss requires probabilities in [0, 1]"
            raise DataValidationError(message)
        probabilities = np.clip(predictions, self.eps, 1.0 - self.eps)
        loss = -np.where(targets == 1, np.log(probabilities), np.log1p(-probabilities))
        return _ErrorState(
            count=state.count + len(targets), total=state.total + float(loss.sum())
        )

    def finalize(self, state: _ErrorState) -> float:
        if state.count == 0:
            message = "cannot score an empty validation source"
            raise DataValidationError(message)
        return state.total / state.count


@attrs.frozen(kw_only=True)
class _R2State:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    squared_error: float = 0.0


@attrs.frozen(kw_only=True)
class R2:
    """Pooled R-squared, using constant-target conventions of zero or one."""

    @property
    def prediction_kind(self) -> Literal["prediction"]:
        return "prediction"

    def initial(self) -> _R2State:
        return _R2State()

    def update(
        self, state: _R2State, targets: FloatArray, predictions: FloatArray
    ) -> _R2State:
        targets, predictions = _metric_arrays(targets, predictions)
        if len(targets) == 0:
            return state
        count = state.count + len(targets)
        mean = float(targets.mean())
        delta = mean - state.mean
        residuals = predictions - targets
        return _R2State(
            count=count,
            mean=state.mean + delta * len(targets) / count,
            m2=state.m2
            + float(np.sum((targets - mean) ** 2))
            + delta * delta * state.count * len(targets) / count,
            squared_error=state.squared_error + float(residuals @ residuals),
        )

    def finalize(self, state: _R2State) -> float:
        if state.count < 2:
            message = "R2 requires at least two validation rows"
            raise DataValidationError(message)
        if state.m2 == 0:
            return 1.0 if state.squared_error == 0 else 0.0
        return 1.0 - state.squared_error / state.m2


@attrs.frozen(kw_only=True)
class Minimize[S]:
    metric: StreamingMetric[S]


@attrs.frozen(kw_only=True)
class Maximize[S]:
    metric: StreamingMetric[S]


@attrs.frozen(kw_only=True)
class Fold:
    train: TrainingSource
    validation: TrainingSource


@attrs.frozen(kw_only=True)
class TrialBudget:
    trials: int
    timeout: float | None = None

    def __attrs_post_init__(self) -> None:
        if (
            isinstance(self.trials, bool)
            or not isinstance(self.trials, int)
            or self.trials <= 0
        ):
            message = "trial count must be a positive integer"
            raise ValidationError(message)
        if self.timeout is not None and (
            not math.isfinite(self.timeout) or self.timeout <= 0
        ):
            message = "tuning timeout must be finite and positive"
            raise ValidationError(message)


def _weights(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(values))


@attrs.frozen(kw_only=True)
class FoldReport:
    fold: int
    score: float | None
    iterations: int
    termination: str
    stationarity: float
    source_passes: int
    failure: FailureReason | None = None


@attrs.frozen(kw_only=True)
class TrialReport:
    number: int
    weights: Mapping[str, float] = attrs.field(converter=_weights)
    status: Literal["complete", "failed"]
    score: float | None
    folds: tuple[FoldReport, ...] = attrs.field(converter=_tuple)


@attrs.frozen(kw_only=True)
class TuningSuccess[M: ModelSpec]:
    best_recipe: Pipeline[M]
    weights: Mapping[str, float] = attrs.field(converter=_weights)
    best_score: float
    trials: tuple[TrialReport, ...] = attrs.field(converter=_tuple)

    @property
    def best_weights(self) -> Mapping[str, float]:
        return self.weights

    @property
    def reports(self) -> tuple[TrialReport, ...]:
        return self.trials


@attrs.frozen(kw_only=True)
class TuningFailure:
    reason: FailureReason
    trials: tuple[TrialReport, ...] = attrs.field(converter=_tuple)

    @property
    def reports(self) -> tuple[TrialReport, ...]:
        return self.trials


type TuningResult[M: ModelSpec] = TuningSuccess[M] | TuningFailure


class _TrialFailed(Exception):
    """Tell Optuna to record an expected numerical trial failure."""


class _Hash(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


def _hash_values(digest: _Hash, values: NDArray) -> None:
    if values.dtype.hasobject:
        for value in values:
            digest.update(
                json.dumps(
                    (type(value).__name__, repr(value)), ensure_ascii=False
                ).encode()
            )
            digest.update(b"\x00")
    else:
        digest.update(values.tobytes())


@attrs.define(kw_only=True)
class _PreparedFold:
    fold: Fold
    session: SolveSession
    validation_fingerprint: str | None = None

    def evaluate[S](
        self,
        predictor: FittedRegressor | FittedBinaryClassifier,
        metric: StreamingMetric[S],
        combined: S,
    ) -> tuple[float, S]:
        source = self.fold.validation
        snapshot = source.snapshot_id
        columns = tuple(source.column_names)
        hashes = {name: hashlib.sha256() for name in columns}
        targets_hash = hashlib.sha256()
        rows_hash = hashlib.sha256()
        state = metric.initial()
        count = 0
        dtypes: tuple[str, ...] | None = None
        iterator = source.batches()
        try:
            for batch in iterator:
                if source.snapshot_id != snapshot:
                    message = "validation source snapshot changed during scoring"
                    raise ReplayError(message)
                if (
                    not isinstance(batch, TrainingBatch)
                    or tuple(batch.columns) != columns
                ):
                    message = (
                        "validation batches must match the declared supervised schema"
                    )
                    raise SchemaError(message)
                current_dtypes = tuple(
                    column.dtype.str for column in batch.columns.values()
                )
                if dtypes is not None and current_dtypes != dtypes:
                    message = "validation column dtypes changed between batches"
                    raise SchemaError(message)
                dtypes = current_dtypes
                count += batch.n_rows
                for name, values in batch.columns.items():
                    _hash_values(hashes[name], values)
                _hash_values(targets_hash, batch.targets)
                _hash_values(rows_hash, batch.row_ids)
                if batch.n_rows == 0:
                    continue
                if metric.prediction_kind == "probability":
                    from stratified_models.predict import FittedBinaryClassifier

                    if not isinstance(predictor, FittedBinaryClassifier):
                        message = "probability metrics require a binary classifier"
                        raise SchemaError(message)
                    predictions = predictor.predict_proba_batch(batch)
                else:
                    predictions = np.asarray(
                        predictor.predict_batch(batch), dtype=np.float64
                    )
                state = metric.update(state, batch.targets, predictions)
                combined = metric.update(combined, batch.targets, predictions)
            if source.snapshot_id != snapshot:
                message = "validation source snapshot changed during scoring"
                raise ReplayError(message)
            if source.n_rows is not None and count != source.n_rows:
                message = f"validation source declared {source.n_rows} rows but yielded {count}"
                raise ReplayError(message)
        finally:
            iterator.close()
        fingerprint = hashlib.sha256(
            repr(
                (
                    snapshot,
                    columns,
                    dtypes,
                    count,
                    tuple(value.hexdigest() for value in hashes.values()),
                    targets_hash.hexdigest(),
                    rows_hash.hexdigest(),
                )
            ).encode()
        ).hexdigest()
        if (
            self.validation_fingerprint is not None
            and fingerprint != self.validation_fingerprint
        ):
            message = "validation source content changed between tuning trials"
            raise ReplayError(message)
        self.validation_fingerprint = fingerprint
        score = float(metric.finalize(state))
        if not math.isfinite(score):
            message = "validation metrics must return finite scores"
            raise DataValidationError(message)
        return score, combined


def _suggest(trial: optuna.Trial, space: WeightSpace) -> dict[str, float]:
    weights = {}
    for name, distribution in space.ranges.items():
        if isinstance(distribution, Choice):
            weights[name] = float(trial.suggest_categorical(name, distribution.values))
        else:
            weights[name] = trial.suggest_float(
                name,
                distribution.low,
                distribution.high,
                log=isinstance(distribution, LogUniform),
            )
    return weights


def _with_weights[M: ModelSpec](
    recipe: Pipeline[M], weights: Mapping[str, float]
) -> Pipeline[M]:
    penalties = tuple(
        attrs.evolve(penalty, weight=weights[penalty.name])
        for penalty in recipe.model.penalties
    )
    return attrs.evolve(recipe, model=attrs.evolve(recipe.model, penalties=penalties))


def tune[M: ModelSpec, S](
    recipe: Pipeline[M],
    *,
    folds: Sequence[Fold],
    search: WeightSpace,
    metric: Minimize[S] | Maximize[S],
    solver: Solver,
    budget: TrialBudget,
    seed: int = 42,
    options: PreparationOptions | None = None,
    cache: SolverCache | None = None,
) -> TuningResult[M]:
    """Select weights using raw folds, then return a recipe without refitting.

    Every fold gets its own fitted transforms and warm-start session. The metric
    pools rows across folds; it does not average unequal batch or fold means.
    The statistics budget is shared equally across prepared folds.
    """
    from stratified_models.fit import FitFailure, SolveSession

    names = {penalty.name for penalty in recipe.model.penalties}
    unknown = set(search.ranges).difference(names)
    if unknown:
        message = f"weight search references unknown penalties: {sorted(unknown)}"
        raise ValidationError(message)
    folds = tuple(folds)
    if not folds:
        message = "tuning requires at least one explicit train/validation fold"
        raise ValidationError(message)
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= np.iinfo(np.uint32).max
    ):
        message = "tuning seed must be an integer in [0, 2**32 - 1]"
        raise ValidationError(message)
    if not isinstance(metric, (Minimize, Maximize)):
        message = "a metric must have an explicit Minimize or Maximize direction"
        raise ValidationError(message)
    if metric.metric.prediction_kind not in {"prediction", "probability"}:
        message = "unsupported streaming metric prediction kind"
        raise ValidationError(message)
    if metric.metric.prediction_kind == "probability" and recipe.model.task != "binary":
        message = "probability metrics require a binary logistic recipe"
        raise SchemaError(message)
    try:
        import optuna
    except ImportError as error:
        message = (
            "Tuning requires the 'tuning' extra: install stratified-models[tuning]."
        )
        raise OptionalDependencyError(message) from error

    shared_cache = SolverCache() if cache is None else cache
    settings = PreparationOptions() if options is None else options
    fold_options = attrs.evolve(
        settings, statistics_max_bytes=settings.statistics_max_bytes // len(folds)
    )
    prepared = tuple(
        _PreparedFold(
            fold=fold,
            session=SolveSession(
                prepare_batches(recipe, fold.train, options=fold_options),
                solver=solver,
                cache=shared_cache,
            ),
        )
        for fold in folds
    )
    base_weights = {penalty.name: penalty.weight for penalty in recipe.model.penalties}
    reports: list[TrialReport] = []

    def objective(trial: optuna.Trial) -> float:
        overrides = _suggest(trial, search)
        weights = {**base_weights, **overrides}
        combined = metric.metric.initial()
        fold_reports: list[FoldReport] = []
        for index, fold in enumerate(prepared):
            result = fold.session.solve(weight_overrides=overrides)
            diagnostics = result.diagnostics
            if isinstance(result, FitFailure):
                fold_reports.append(
                    FoldReport(
                        fold=index,
                        score=None,
                        iterations=diagnostics.iterations,
                        termination=diagnostics.termination,
                        stationarity=diagnostics.stationarity,
                        source_passes=diagnostics.source_passes,
                        failure=result.reason,
                    )
                )
                reports.append(
                    TrialReport(
                        number=trial.number,
                        weights=weights,
                        status="failed",
                        score=None,
                        folds=tuple(fold_reports),
                    )
                )
                raise _TrialFailed(result.reason.message)
            score, combined = fold.evaluate(result.predictor, metric.metric, combined)
            fold_reports.append(
                FoldReport(
                    fold=index,
                    score=score,
                    iterations=diagnostics.iterations,
                    termination=diagnostics.termination,
                    stationarity=diagnostics.stationarity,
                    source_passes=diagnostics.source_passes,
                )
            )
        score = float(metric.metric.finalize(combined))
        if not math.isfinite(score):
            message = "validation metrics must return finite scores"
            raise DataValidationError(message)
        reports.append(
            TrialReport(
                number=trial.number,
                weights=weights,
                status="complete",
                score=score,
                folds=tuple(fold_reports),
            )
        )
        return score

    study = optuna.create_study(
        direction="minimize" if isinstance(metric, Minimize) else "maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective,
        n_trials=budget.trials,
        timeout=budget.timeout,
        catch=(_TrialFailed,),
        n_jobs=1,
        show_progress_bar=False,
    )
    complete = tuple(report for report in reports if report.status == "complete")
    if not complete:
        return TuningFailure(
            reason=FailureReason(
                code="no_successful_trials",
                message="No tuning trial converged on every validation fold.",
            ),
            trials=tuple(reports),
        )
    best = (
        min(
            complete,
            key=lambda report: float(report.score)
            if report.score is not None
            else math.inf,
        )
        if isinstance(metric, Minimize)
        else max(
            complete,
            key=lambda report: float(report.score)
            if report.score is not None
            else -math.inf,
        )
    )
    assert best.score is not None
    return TuningSuccess(
        best_recipe=_with_weights(recipe, best.weights),
        weights=best.weights,
        best_score=best.score,
        trials=tuple(reports),
    )
