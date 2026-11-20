"""Immutable task-specific fitted predictors and bounded prediction streams."""

from __future__ import annotations

from collections.abc import Generator, Iterator, Mapping, Sequence
from math import prod
from types import MappingProxyType, TracebackType
from typing import Literal, Protocol, Self

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from stratified_models.data import (
    Array,
    FeatureBatch,
    FeatureSource,
    TrainingBatch,
    TrainingSource,
    feature_source_from_data,
    readonly_array,
    readonly_float_array,
)
from stratified_models.errors import DataValidationError, ReplayError, SchemaError
from stratified_models.prepared import coordinate_key, encode_nodes
from stratified_models.preprocessing import FittedTransformChain
from stratified_models.solvers.types import Diagnostics
from stratified_models.specs import (
    Axis,
    BinaryLogisticSpec,
    ModelSpec,
    Pipeline,
    RegressionSpec,
)


def _weights(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType({name: float(value) for name, value in values.items()})


def _validate(
    model: ModelSpec, coefficients: NDArray[np.float64], weights: Mapping[str, float]
) -> None:
    expected = (prod(axis.size for axis in model.axes), len(model.features))
    if coefficients.shape != expected or not np.isfinite(coefficients).all():
        raise DataValidationError(
            f"finite coefficients with shape {expected} are required"
        )
    if set(weights) != {penalty.name for penalty in model.penalties}:
        raise SchemaError(
            "fitted penalty weights must match the model's declared penalties"
        )
    if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise DataValidationError(
            "fitted penalty weights must be finite and nonnegative"
        )


class _Predictor(Protocol):
    @property
    def features(self) -> tuple[str, ...]: ...

    @property
    def axes(self) -> tuple[Axis, ...]: ...

    @property
    def transformers(self) -> FittedTransformChain: ...

    @property
    def coefficients(self) -> NDArray[np.float64]: ...


def _scores(
    predictor: _Predictor, raw: FeatureBatch | TrainingBatch
) -> NDArray[np.float64]:
    features = raw.features() if isinstance(raw, TrainingBatch) else raw
    batch = predictor.transformers.transform(features)
    missing = set(predictor.features).difference(batch.columns)
    if missing:
        raise SchemaError(f"missing prediction features: {sorted(missing)}")
    try:
        x = np.column_stack(
            [
                np.asarray(batch.columns[name], dtype=np.float64)
                for name in predictor.features
            ]
        )
    except (TypeError, ValueError) as error:
        raise DataValidationError(
            "prediction features must be real numerical columns"
        ) from error
    if not np.isfinite(x).all():
        raise DataValidationError("prediction features must be finite")
    nodes = encode_nodes(batch.columns, predictor.axes, batch.n_rows)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = np.sum(x * predictor.coefficients[nodes], axis=1)
    if not np.isfinite(scores).all():
        raise DataValidationError(
            "prediction arithmetic exceeded finite float64 values"
        )
    return scores


@attrs.frozen(kw_only=True, eq=False)
class PredictionBatch:
    row_ids: Array = attrs.field(converter=readonly_array)
    values: Array = attrs.field(converter=readonly_array)

    def __attrs_post_init__(self) -> None:
        if (
            self.row_ids.ndim != 1
            or self.values.ndim not in (1, 2)
            or len(self.row_ids) != len(self.values)
        ):
            raise SchemaError("prediction values and row identifiers must align")


class PredictionStream(Iterator[PredictionBatch]):
    """A closable stream; context management releases an early-stopped reader."""

    def __init__(self, iterator: Generator[PredictionBatch]) -> None:
        self._iterator = iterator

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> PredictionBatch:
        return next(self._iterator)

    def close(self) -> None:
        self._iterator.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False


def _stream(
    predictor: _Predictor,
    source: FeatureSource | TrainingSource,
    kind: Literal["score", "probability", "label"],
    threshold: float = 0.5,
) -> Generator[PredictionBatch]:
    snapshot = source.snapshot_id
    columns = tuple(source.column_names)
    count = 0
    iterator = source.batches()
    try:
        for batch in iterator:
            if source.snapshot_id != snapshot:
                raise ReplayError("prediction source changed during reading")
            if not isinstance(batch, (FeatureBatch, TrainingBatch)):
                raise SchemaError(
                    "prediction sources must yield feature or training batches"
                )
            if tuple(source.column_names) != columns or tuple(batch.columns) != columns:
                raise SchemaError(
                    "prediction batch columns differ from the declared source schema"
                )
            scores = _scores(predictor, batch)
            if kind == "probability":
                positive = expit(scores)
                values = np.column_stack((1.0 - positive, positive))
            elif kind == "label":
                values = (expit(scores) >= threshold).astype(np.int64)
            else:
                values = scores
            count += len(batch.row_ids)
            yield PredictionBatch(row_ids=batch.row_ids, values=values)
        if (
            source.snapshot_id != snapshot
            or source.n_rows is not None
            and count != source.n_rows
        ):
            raise ReplayError("prediction source differs from its declared snapshot")
    finally:
        iterator.close()


def _in_memory(
    predictor: _Predictor,
    X: object,
    kind: Literal["score", "probability", "label"],
    threshold: float = 0.5,
    feature_names: Sequence[str] | None = None,
) -> NDArray:
    source = feature_source_from_data(X, feature_names=feature_names)
    iterator = _stream(predictor, source, kind, threshold)
    try:
        chunks = [batch.values for batch in iterator]
    finally:
        iterator.close()
    if not chunks:
        if kind == "probability":
            return np.empty((0, 2))
        return np.empty(0, dtype=np.int64 if kind == "label" else np.float64)
    return np.concatenate(chunks)


def _at_node(
    predictor: _Predictor, node: Mapping[str, str | int]
) -> NDArray[np.float64]:
    if set(node) != {axis.name for axis in predictor.axes}:
        raise SchemaError("node coordinates must name every declared axis exactly once")
    index = (
        int(
            np.ravel_multi_index(
                tuple(axis.index(node[axis.name]) for axis in predictor.axes),
                tuple(axis.size for axis in predictor.axes),
            )
        )
        if predictor.axes
        else 0
    )
    return predictor.coefficients[index]


@attrs.frozen(kw_only=True, eq=False)
class FittedRegressor:
    pipeline: Pipeline[RegressionSpec]
    transformers: FittedTransformChain
    coefficients: NDArray[np.float64] = attrs.field(converter=readonly_float_array)
    weights: Mapping[str, float] = attrs.field(converter=_weights)
    training_diagnostics: Diagnostics | None = None

    def __attrs_post_init__(self) -> None:
        if not isinstance(self.pipeline.model, RegressionSpec):
            raise SchemaError("a fitted regressor requires a regression specification")
        _validate(self.pipeline.model, self.coefficients, self.weights)

    @property
    def features(self) -> tuple[str, ...]:
        return self.pipeline.model.features

    @property
    def axes(self) -> tuple[Axis, ...]:
        return self.pipeline.model.axes

    @property
    def coordinates(self) -> str:
        return coordinate_key(self.pipeline.model, self.transformers)

    def predict(
        self, X: object, *, feature_names: Sequence[str] | None = None
    ) -> NDArray[np.float64]:
        return _in_memory(self, X, "score", feature_names=feature_names)

    def predict_batch(self, batch: FeatureBatch | TrainingBatch) -> NDArray[np.float64]:
        return _scores(self, batch)

    def predict_batches(
        self, source: FeatureSource | TrainingSource
    ) -> PredictionStream:
        return PredictionStream(_stream(self, source, "score"))

    def coefficients_at(self, node: Mapping[str, str | int]) -> NDArray[np.float64]:
        return _at_node(self, node)


@attrs.frozen(kw_only=True, eq=False)
class FittedBinaryClassifier:
    pipeline: Pipeline[BinaryLogisticSpec]
    transformers: FittedTransformChain
    coefficients: NDArray[np.float64] = attrs.field(converter=readonly_float_array)
    weights: Mapping[str, float] = attrs.field(converter=_weights)
    training_diagnostics: Diagnostics | None = None

    def __attrs_post_init__(self) -> None:
        if not isinstance(self.pipeline.model, BinaryLogisticSpec):
            raise SchemaError(
                "a fitted binary classifier requires a logistic specification"
            )
        _validate(self.pipeline.model, self.coefficients, self.weights)

    @property
    def features(self) -> tuple[str, ...]:
        return self.pipeline.model.features

    @property
    def axes(self) -> tuple[Axis, ...]:
        return self.pipeline.model.axes

    @property
    def classes(self) -> tuple[int, int]:
        return (0, 1)

    @property
    def coordinates(self) -> str:
        return coordinate_key(self.pipeline.model, self.transformers)

    def decision_function(
        self, X: object, *, feature_names: Sequence[str] | None = None
    ) -> NDArray[np.float64]:
        return _in_memory(self, X, "score", feature_names=feature_names)

    def predict_proba(
        self, X: object, *, feature_names: Sequence[str] | None = None
    ) -> NDArray[np.float64]:
        return _in_memory(self, X, "probability", feature_names=feature_names)

    def predict(
        self,
        X: object,
        *,
        threshold: float = 0.5,
        feature_names: Sequence[str] | None = None,
    ) -> NDArray[np.int64]:
        _validate_threshold(threshold)
        return _in_memory(self, X, "label", threshold, feature_names)

    def predict_batch(
        self, batch: FeatureBatch | TrainingBatch, *, threshold: float = 0.5
    ) -> NDArray[np.int64]:
        _validate_threshold(threshold)
        return (expit(_scores(self, batch)) >= threshold).astype(np.int64)

    def predict_proba_batch(
        self, batch: FeatureBatch | TrainingBatch
    ) -> NDArray[np.float64]:
        positive = expit(_scores(self, batch))
        return np.column_stack((1.0 - positive, positive))

    def predict_batches(
        self, source: FeatureSource | TrainingSource, *, threshold: float = 0.5
    ) -> PredictionStream:
        _validate_threshold(threshold)
        return PredictionStream(_stream(self, source, "label", threshold))

    def predict_proba_batches(
        self, source: FeatureSource | TrainingSource
    ) -> PredictionStream:
        return PredictionStream(_stream(self, source, "probability"))

    def coefficients_at(self, node: Mapping[str, str | int]) -> NDArray[np.float64]:
        return _at_node(self, node)


def _validate_threshold(threshold: float) -> None:
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise DataValidationError(
            "classification threshold must be between zero and one"
        )


type FittedPredictor = FittedRegressor | FittedBinaryClassifier
