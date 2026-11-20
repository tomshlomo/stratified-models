"""Explicit fitted transforms for replayable batches.

Fixed-width grids cover ``[origin, origin + width * n_bins)``. Standardization
uses population variance, with unit scale for columns constant in training.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import closing
from typing import Protocol, runtime_checkable

import attrs
import numpy as np
from numpy.typing import NDArray

from stratified_models.data import (
    BatchSource,
    FeatureBatch,
    TrainingBatch,
    readonly_float_array,
)
from stratified_models.errors import (
    DataValidationError,
    ReplayError,
    SchemaError,
    ValidationError,
)
from stratified_models.specs import validate_name, validate_names

type Batch = FeatureBatch | TrainingBatch


def _tuple[T](values: Iterable[T]) -> tuple[T, ...]:
    return tuple(values)


def _digest(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _require_columns(required: Sequence[str], columns: Sequence[str]) -> None:
    missing = set(required).difference(columns)
    if missing:
        message = f"missing transform columns: {sorted(missing)}"
        raise SchemaError(message)


def _add_column(columns: Sequence[str], name: str) -> tuple[str, ...]:
    if name in columns:
        message = f"transform output {name!r} already exists"
        raise SchemaError(message)
    return (*columns, name)


def _numeric_column(batch: Batch, name: str) -> NDArray[np.float64]:
    _require_columns((name,), tuple(batch.columns))
    try:
        values = np.asarray(batch.columns[name], dtype=np.float64)
    except (ValueError, TypeError) as error:
        message = f"column {name!r} must be numerical"
        raise DataValidationError(message) from error
    if not np.all(np.isfinite(values)):
        message = f"column {name!r} contains nonfinite values"
        raise DataValidationError(message)
    return values


def _replace_columns[B: Batch](batch: B, columns: Mapping[str, NDArray]) -> B:
    return attrs.evolve(batch, columns=columns)


@runtime_checkable
class FittedTransform(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def output_columns(self, columns: Sequence[str]) -> tuple[str, ...]: ...

    def transform[B: Batch](self, batch: B) -> B: ...


@runtime_checkable
class TransformRecipe(Protocol):
    def fit(self, source: BatchSource[Batch]) -> FittedTransform: ...


type TransformStep = TransformRecipe | FittedTransform


def _scan(source: BatchSource[Batch]) -> Generator[Batch]:
    snapshot = source.snapshot_id
    count = 0
    iterator = source.batches()
    try:
        for batch in iterator:
            if source.snapshot_id != snapshot:
                message = "source snapshot changed during preprocessing"
                raise ReplayError(message)
            count += len(batch.row_ids)
            yield batch
        if source.snapshot_id != snapshot:
            message = "source snapshot changed during preprocessing"
            raise ReplayError(message)
        if source.n_rows is not None and count != source.n_rows:
            message = f"source declared {source.n_rows} rows but replayed {count}"
            raise ReplayError(message)
    finally:
        iterator.close()


@attrs.frozen(kw_only=True)
class Standardize:
    columns: tuple[str, ...] = attrs.field(converter=_tuple)

    def __attrs_post_init__(self) -> None:
        validate_names(self.columns, kind="standardization columns")

    def fit(self, source: BatchSource[Batch]) -> FittedStandardize:
        _require_columns(self.columns, source.column_names)
        count = 0
        means = np.zeros(len(self.columns), dtype=np.float64)
        m2 = np.zeros(len(self.columns), dtype=np.float64)
        with closing(_scan(source)) as batches:
            for batch in batches:
                values = np.column_stack(
                    [_numeric_column(batch, name) for name in self.columns]
                )
                batch_count = len(values)
                if batch_count == 0:
                    continue
                batch_means = values.mean(axis=0)
                centered = values - batch_means
                batch_m2 = np.sum(centered * centered, axis=0)
                delta = batch_means - means
                total = count + batch_count
                means += delta * (batch_count / total)
                m2 += batch_m2 + delta * delta * (count * batch_count / total)
                count = total
        if count == 0:
            message = "cannot fit standardization to an empty source"
            raise DataValidationError(message)
        scales = np.sqrt(np.maximum(m2 / count, 0.0))
        scales[scales == 0] = 1.0
        return FittedStandardize(columns=self.columns, means=means, scales=scales)


@attrs.frozen(kw_only=True)
class FittedStandardize:
    columns: tuple[str, ...] = attrs.field(converter=_tuple)
    means: NDArray[np.float64] = attrs.field(
        converter=readonly_float_array,
        eq=attrs.cmp_using(eq=np.array_equal),
        hash=False,
    )
    scales: NDArray[np.float64] = attrs.field(
        converter=readonly_float_array,
        eq=attrs.cmp_using(eq=np.array_equal),
        hash=False,
    )

    def __attrs_post_init__(self) -> None:
        validate_names(self.columns, kind="standardization columns")
        if (
            self.means.shape != (len(self.columns),)
            or self.scales.shape != self.means.shape
        ):
            message = "standardization parameters must match the column count"
            raise ValidationError(message)
        if (
            not np.all(np.isfinite(self.means))
            or not np.all(np.isfinite(self.scales))
            or np.any(self.scales <= 0)
        ):
            message = "standardization requires finite means and positive finite scales"
            raise ValidationError(message)

    @property
    def fingerprint(self) -> str:
        return _digest(
            (
                b"standardize-v1",
                repr(self.columns).encode(),
                self.means.tobytes(),
                self.scales.tobytes(),
            )
        )

    def output_columns(self, columns: Sequence[str]) -> tuple[str, ...]:
        _require_columns(self.columns, columns)
        return tuple(columns)

    def transform[B: Batch](self, batch: B) -> B:
        self.output_columns(tuple(batch.columns))
        columns = dict(batch.columns)
        for index, name in enumerate(self.columns):
            columns[name] = (
                _numeric_column(batch, name) - self.means[index]
            ) / self.scales[index]
        return _replace_columns(batch, columns)


@attrs.frozen(kw_only=True)
class AddIntercept:
    name: str

    def __attrs_post_init__(self) -> None:
        validate_name(self.name, kind="intercept name")

    def fit(self, source: BatchSource[Batch]) -> FittedAddIntercept:
        fitted = FittedAddIntercept(name=self.name)
        fitted.output_columns(source.column_names)
        return fitted


@attrs.frozen(kw_only=True)
class FittedAddIntercept:
    name: str

    def __attrs_post_init__(self) -> None:
        validate_name(self.name, kind="intercept name")

    @property
    def fingerprint(self) -> str:
        return _digest((b"intercept-v1", self.name.encode()))

    def output_columns(self, columns: Sequence[str]) -> tuple[str, ...]:
        return _add_column(columns, self.name)

    def transform[B: Batch](self, batch: B) -> B:
        self.output_columns(tuple(batch.columns))
        return _replace_columns(
            batch, {**batch.columns, self.name: np.ones(len(batch.row_ids))}
        )


def _validate_bins(
    column: str, output: str, origin: float, width: float, n_bins: int
) -> None:
    validate_name(column, kind="bin input column")
    validate_name(output, kind="bin output column")
    if column == output:
        message = "bin output must differ from its input column"
        raise ValidationError(message)
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        message = "n_bins must be a positive integer"
        raise ValidationError(message)
    if not math.isfinite(origin) or not math.isfinite(width) or width <= 0:
        message = "bin origin must be finite and width must be positive and finite"
        raise ValidationError(message)
    if not math.isfinite(origin + width * n_bins) or origin + width * n_bins <= origin:
        message = "bin domain must have finite distinct bounds"
        raise ValidationError(message)


@attrs.frozen(kw_only=True)
class FixedWidthBins:
    column: str
    output: str
    origin: float = attrs.field(converter=float)
    width: float = attrs.field(converter=float)
    n_bins: int

    def __attrs_post_init__(self) -> None:
        _validate_bins(self.column, self.output, self.origin, self.width, self.n_bins)

    def fit(self, source: BatchSource[Batch]) -> FittedFixedWidthBins:
        fitted = FittedFixedWidthBins(
            column=self.column,
            output=self.output,
            origin=self.origin,
            width=self.width,
            n_bins=self.n_bins,
        )
        fitted.output_columns(source.column_names)
        return fitted


@attrs.frozen(kw_only=True)
class FittedFixedWidthBins:
    column: str
    output: str
    origin: float = attrs.field(converter=float)
    width: float = attrs.field(converter=float)
    n_bins: int

    def __attrs_post_init__(self) -> None:
        _validate_bins(self.column, self.output, self.origin, self.width, self.n_bins)

    @property
    def fingerprint(self) -> str:
        return _digest(
            (
                b"fixed-bins-v1",
                repr(
                    (self.column, self.output, self.origin, self.width, self.n_bins)
                ).encode(),
            )
        )

    def output_columns(self, columns: Sequence[str]) -> tuple[str, ...]:
        _require_columns((self.column,), columns)
        return _add_column(columns, self.output)

    def transform[B: Batch](self, batch: B) -> B:
        self.output_columns(tuple(batch.columns))
        values = _numeric_column(batch, self.column)
        upper = self.origin + self.width * self.n_bins
        if np.any(values < self.origin) or np.any(values >= upper):
            message = f"column {self.column!r} contains values outside bin domain [{self.origin}, {upper})"
            raise DataValidationError(message)
        bins = np.minimum(
            np.floor((values - self.origin) / self.width).astype(np.int64),
            self.n_bins - 1,
        )
        return _replace_columns(batch, {**batch.columns, self.output: bins})


@attrs.frozen(kw_only=True)
class TransformChain:
    steps: tuple[TransformStep, ...] = attrs.field(converter=_tuple, factory=tuple)

    def __attrs_post_init__(self) -> None:
        if any(
            not isinstance(step, (FittedTransform, TransformRecipe))
            for step in self.steps
        ):
            message = (
                "transform steps must implement fit or the fitted-transform protocol"
            )
            raise ValidationError(message)

    def fit(self, source: BatchSource[Batch]) -> FittedTransformChain:
        fitted_steps: list[FittedTransform] = []
        current: BatchSource[Batch] = source
        for step in self.steps:
            fitted = step if isinstance(step, FittedTransform) else step.fit(current)
            fitted.output_columns(current.column_names)
            fitted_steps.append(fitted)
            current = TransformedSource(
                source=current, transformers=FittedTransformChain(steps=(fitted,))
            )
        return FittedTransformChain(steps=tuple(fitted_steps))

    def fit_batches(self, source: BatchSource[Batch]) -> FittedTransformChain:
        return self.fit(source)


@attrs.frozen(kw_only=True)
class FittedTransformChain:
    steps: tuple[FittedTransform, ...] = attrs.field(converter=_tuple, factory=tuple)

    def __attrs_post_init__(self) -> None:
        if any(not isinstance(step, FittedTransform) for step in self.steps):
            message = "fitted steps must implement the fitted-transform protocol"
            raise ValidationError(message)

    @property
    def fingerprint(self) -> str:
        return _digest(
            (b"chain-v1", *(step.fingerprint.encode() for step in self.steps))
        )

    def output_columns(self, columns: Sequence[str]) -> tuple[str, ...]:
        output = tuple(columns)
        for step in self.steps:
            output = step.output_columns(output)
        return output

    def transform[B: Batch](self, batch: B) -> B:
        transformed = batch
        for step in self.steps:
            expected_columns = step.output_columns(tuple(transformed.columns))
            transformed = step.transform(transformed)
            if type(transformed) is not type(batch):
                message = "feature transforms must preserve the batch kind"
                raise SchemaError(message)
            if tuple(transformed.columns) != expected_columns:
                message = "feature transform output does not match its declared columns"
                raise SchemaError(message)
            if not np.array_equal(batch.row_ids, transformed.row_ids):
                message = "feature transforms must preserve row identities and order"
                raise SchemaError(message)
            if isinstance(batch, TrainingBatch) and (
                not isinstance(transformed, TrainingBatch)
                or not np.array_equal(batch.targets, transformed.targets)
            ):
                message = "feature transforms must preserve training targets"
                raise SchemaError(message)
        return transformed

    def wrap_source[B: Batch](self, source: BatchSource[B]) -> TransformedSource[B]:
        self.output_columns(source.column_names)
        return TransformedSource(source=source, transformers=self)


@attrs.frozen(kw_only=True)
class TransformedSource[B: Batch]:
    source: BatchSource[B]
    transformers: FittedTransformChain

    @property
    def snapshot_id(self) -> str:
        return _digest(
            (self.source.snapshot_id.encode(), self.transformers.fingerprint.encode())
        )

    @property
    def n_rows(self) -> int | None:
        return self.source.n_rows

    @property
    def column_names(self) -> tuple[str, ...]:
        return self.transformers.output_columns(self.source.column_names)

    def batches(self) -> Generator[B]:
        iterator = self.source.batches()
        try:
            for batch in iterator:
                yield self.transformers.transform(batch)
        finally:
            iterator.close()


def compose(*steps: TransformStep) -> TransformChain:
    return TransformChain(steps=steps)
