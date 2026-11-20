"""Immutable preparation of one replayable training-data snapshot."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Generator, Mapping, Sequence
from typing import Literal, Protocol

import attrs
import numpy as np

from stratified_models.data import (
    Array,
    NumericBatch,
    TrainingBatch,
    TrainingSource,
    source_from_data,
)
from stratified_models.errors import DataValidationError, ReplayError, SchemaError
from stratified_models.preprocessing import FittedTransformChain
from stratified_models.solvers.types import QuadraticStatistics
from stratified_models.specs import Axis, ModelSpec, Penalty, Pipeline


@attrs.frozen(kw_only=True)
class PreparationOptions:
    batch_size: int = 8192
    statistics_max_bytes: int = 128 * 1024 * 1024
    max_model_bytes: int = 512 * 1024 * 1024

    def __attrs_post_init__(self) -> None:
        for name in ("batch_size", "statistics_max_bytes", "max_model_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.batch_size == 0 or self.max_model_bytes == 0:
            raise ValueError("batch_size and max_model_bytes must be positive")


@attrs.frozen(kw_only=True)
class PreparationDiagnostics:
    source_passes: int
    rows_scanned: int
    elapsed_seconds: float
    statistics_bytes: int
    coefficient_count: int


class _Hash(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


def _schema_bytes(model: ModelSpec) -> bytes:
    return json.dumps(
        {
            "task": model.task,
            "features": model.features,
            "axes": [(axis.name, axis.nodes) for axis in model.axes],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def coordinate_key(model: ModelSpec, transformers: FittedTransformChain) -> str:
    return hashlib.sha256(
        _schema_bytes(model) + transformers.fingerprint.encode()
    ).hexdigest()


def encode_nodes(
    columns: Mapping[str, Array], axes: tuple[Axis, ...], n_rows: int
) -> np.ndarray:
    # Source batches expose an immutable Mapping; keep this helper usable by
    # prediction without depending on a dataframe engine.
    if not axes:
        return np.zeros(n_rows, dtype=np.int64)
    codes = []
    for axis in axes:
        if axis.name not in columns:
            raise SchemaError(f"missing stratum column {axis.name!r}")
        codes.append(axis.encode(columns[axis.name]))
    return np.ravel_multi_index(tuple(codes), tuple(a.size for a in axes))


def _numeric_batches(
    source: TrainingSource,
    transformers: FittedTransformChain,
    model: ModelSpec,
    snapshot: str,
) -> Generator[NumericBatch]:
    expected_columns = tuple(source.column_names)
    iterator = source.batches()
    try:
        for raw in iterator:
            if source.snapshot_id != snapshot:
                raise ReplayError("source snapshot changed while reading training data")
            if not isinstance(raw, TrainingBatch):
                raise SchemaError("training sources must yield TrainingBatch values")
            if tuple(raw.columns) != expected_columns:
                raise SchemaError(
                    "batch columns do not match the declared source schema"
                )
            batch = transformers.transform(raw)
            missing = set(model.features).difference(batch.columns)
            if missing:
                raise SchemaError(f"missing regression features: {sorted(missing)}")
            try:
                x = np.column_stack(
                    [
                        np.asarray(batch.columns[name], dtype=np.float64)
                        for name in model.features
                    ]
                )
            except (TypeError, ValueError) as error:
                raise DataValidationError(
                    "regression features must be real numerical columns"
                ) from error
            if model.task == "binary" and not np.all(
                (batch.targets == 0) | (batch.targets == 1)
            ):
                raise DataValidationError(
                    "binary logistic targets must be encoded as zero or one"
                )
            nodes = encode_nodes(batch.columns, model.axes, batch.n_rows)
            if batch.n_rows:
                yield NumericBatch(
                    x=x, y=batch.targets, nodes=nodes, row_ids=batch.row_ids
                )
        if source.snapshot_id != snapshot:
            raise ReplayError("source snapshot changed while reading training data")
    finally:
        iterator.close()


def _update_hashes(
    batch: NumericBatch, x_hash: _Hash, y_hash: _Hash, nodes_hash: _Hash
) -> None:
    # Independent byte streams make fingerprints independent of batch boundaries.
    x_hash.update(batch.x.tobytes(order="C"))
    y_hash.update(batch.y.tobytes(order="C"))
    nodes_hash.update(batch.nodes.tobytes(order="C"))


def _fingerprints(
    model: ModelSpec, x_hash: _Hash, y_hash: _Hash, nodes_hash: _Hash, count: int
) -> tuple[str, str]:
    common = (
        _schema_bytes(model)
        + str(count).encode()
        + x_hash.hexdigest().encode()
        + nodes_hash.hexdigest().encode()
    )
    design = hashlib.sha256(common).hexdigest()
    full = hashlib.sha256(common + y_hash.hexdigest().encode()).hexdigest()
    return full, design


@attrs.frozen(kw_only=True, eq=False)
class PreparedProblem[M: ModelSpec]:
    pipeline: Pipeline[M]
    transformers: FittedTransformChain
    source: TrainingSource
    snapshot_id: str
    n_samples: int
    fingerprint: str
    design_fingerprint: str
    statistics: QuadraticStatistics | None
    options: PreparationOptions
    preparation_passes: int
    preparation_rows: int
    preparation_seconds: float = 0.0

    @property
    def task(self) -> Literal["regression", "binary"]:
        return self.pipeline.model.task

    @property
    def axes(self) -> tuple[Axis, ...]:
        return self.pipeline.model.axes

    @property
    def axis_sizes(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)

    @property
    def features(self) -> tuple[str, ...]:
        return self.pipeline.model.features

    @property
    def penalties(self) -> tuple[Penalty, ...]:
        return self.pipeline.model.penalties

    @property
    def n_nodes(self) -> int:
        return math.prod(self.axis_sizes)

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def coordinates(self) -> str:
        return coordinate_key(self.pipeline.model, self.transformers)

    @property
    def preparation(self) -> PreparationDiagnostics:
        statistics_bytes = (
            0
            if self.statistics is None
            else self.statistics.xtx.nbytes + self.statistics.xty.nbytes + 8
        )
        return PreparationDiagnostics(
            source_passes=self.preparation_passes,
            rows_scanned=self.preparation_rows,
            elapsed_seconds=self.preparation_seconds,
            statistics_bytes=statistics_bytes,
            coefficient_count=self.n_nodes * self.n_features,
        )

    def iter_numeric(self) -> Generator[NumericBatch]:
        x_hash, y_hash, nodes_hash = (hashlib.sha256() for _ in range(3))
        count = 0
        iterator = _numeric_batches(
            self.source, self.transformers, self.pipeline.model, self.snapshot_id
        )
        try:
            for batch in iterator:
                count += batch.n_rows
                _update_hashes(batch, x_hash, y_hash, nodes_hash)
                yield batch
            full, _ = _fingerprints(
                self.pipeline.model, x_hash, y_hash, nodes_hash, count
            )
            if count != self.n_samples or full != self.fingerprint:
                raise ReplayError(
                    "training source content differs from its prepared snapshot"
                )
        finally:
            iterator.close()


class _CountedSource:
    def __init__(self, source: TrainingSource) -> None:
        self.source = source
        self.passes = 0
        self.rows = 0

    @property
    def snapshot_id(self) -> str:
        return self.source.snapshot_id

    @property
    def n_rows(self) -> int | None:
        return self.source.n_rows

    @property
    def column_names(self) -> tuple[str, ...]:
        return self.source.column_names

    def batches(self) -> Generator[TrainingBatch]:
        self.passes += 1
        iterator = self.source.batches()
        try:
            for batch in iterator:
                self.rows += batch.n_rows
                yield batch
        finally:
            iterator.close()


def prepare_batches[M: ModelSpec](
    pipeline: Pipeline[M],
    source: TrainingSource,
    *,
    options: PreparationOptions | None = None,
) -> PreparedProblem[M]:
    started = time.perf_counter()
    settings = PreparationOptions() if options is None else options
    n_nodes = math.prod(axis.size for axis in pipeline.model.axes)
    n_features = len(pipeline.model.features)
    # ADMM holds a constant number of coefficient-sized arrays and work vectors.
    if 16 * n_nodes * n_features * 8 > settings.max_model_bytes:
        raise DataValidationError("coefficient work state exceeds max_model_bytes")
    snapshot = source.snapshot_id
    monitored = _CountedSource(source)
    transformers = pipeline.preprocessing.fit(monitored)
    output = transformers.output_columns(source.column_names)
    missing = set(
        (*pipeline.model.features, *(a.name for a in pipeline.model.axes))
    ).difference(output)
    if missing:
        raise SchemaError(f"prepared pipeline is missing columns: {sorted(missing)}")
    stats_bytes = 3 * 8 * (n_nodes * n_features * n_features + n_nodes * n_features + 1)
    use_stats = (
        pipeline.model.task == "regression"
        and stats_bytes <= settings.statistics_max_bytes
    )
    xtx = np.zeros((n_nodes, n_features, n_features)) if use_stats else None
    xty = np.zeros((n_nodes, n_features)) if use_stats else None
    yty = 0.0
    x_hash, y_hash, nodes_hash = (hashlib.sha256() for _ in range(3))
    count = 0
    iterator = _numeric_batches(source, transformers, pipeline.model, snapshot)
    try:
        for batch in iterator:
            count += batch.n_rows
            _update_hashes(batch, x_hash, y_hash, nodes_hash)
            if xtx is not None and xty is not None:
                for node in np.unique(batch.nodes):
                    selected = batch.nodes == node
                    block, target = batch.x[selected], batch.y[selected]
                    xtx[node] += block.T @ block
                    xty[node] += block.T @ target
                yty += float(batch.y @ batch.y)
    finally:
        iterator.close()
    if count == 0:
        raise DataValidationError("fitting requires at least one training sample")
    if source.n_rows is not None and count != source.n_rows:
        raise ReplayError(f"source declared {source.n_rows} rows but yielded {count}")
    full, design = _fingerprints(pipeline.model, x_hash, y_hash, nodes_hash, count)
    statistics = (
        None
        if xtx is None or xty is None
        else QuadraticStatistics(xtx=xtx, xty=xty, yty=yty)
    )
    return PreparedProblem(
        pipeline=pipeline,
        transformers=transformers,
        source=source,
        snapshot_id=snapshot,
        n_samples=count,
        fingerprint=full,
        design_fingerprint=design,
        statistics=statistics,
        options=settings,
        preparation_passes=monitored.passes + 1,
        preparation_rows=monitored.rows + count,
        preparation_seconds=time.perf_counter() - started,
    )


def prepare[M: ModelSpec](
    pipeline: Pipeline[M],
    X: object,
    y: object,
    *,
    options: PreparationOptions | None = None,
    feature_names: Sequence[str] | None = None,
) -> PreparedProblem[M]:
    settings = PreparationOptions() if options is None else options
    source = source_from_data(
        X, y, batch_size=settings.batch_size, feature_names=feature_names
    )
    return prepare_batches(pipeline, source, options=settings)
