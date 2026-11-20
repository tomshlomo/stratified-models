"""Owned batches and replayable, engine-independent data sources.

The optional dataframe and Parquet integrations are imported only by their
adapters. Numerical code consumes NumPy batches and need not depend on them.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Generator, Iterator, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Protocol, cast, overload, runtime_checkable

import attrs
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from stratified_models.errors import (
    DataValidationError,
    OptionalDependencyError,
    ReplayError,
    SchemaError,
)

type Array = NDArray[np.generic]
type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]

_DEFAULT_BATCH_SIZE = 4096


def readonly_array(value: ArrayLike, *, dtype: DTypeLike | None = None) -> Array:
    """Copy an array, preventing mutation through the caller's original buffer."""
    result = np.array(value, dtype=dtype, copy=True, order="C", subok=False)
    if result.dtype.hasobject:
        for item in result.flat:
            if not _is_immutable_scalar(item):
                msg = "Object arrays may contain only immutable scalar values."
                raise DataValidationError(msg)
        result.setflags(write=False)
        return result.view()
    # An immutable bytes owner also prevents re-enabling the writeable flag.
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


def readonly_float_array(value: ArrayLike) -> FloatArray:
    try:
        if _has_complex(value):
            msg = "Expected real numerical values, not complex values."
            raise DataValidationError(msg)
        return cast("FloatArray", readonly_array(value, dtype=np.float64))
    except (TypeError, ValueError) as error:
        msg = "Expected an array of real numerical values."
        raise DataValidationError(msg) from error


def _has_complex(value: ArrayLike) -> bool:
    values = np.asarray(value)
    return bool(np.iscomplexobj(values)) or (
        values.dtype.hasobject
        and any(isinstance(item, (complex, np.complexfloating)) for item in values.flat)
    )


def _is_immutable_scalar(value: object) -> bool:
    if value is None or isinstance(
        value,
        (str, bytes, int, float, bool, complex, np.generic, date, datetime, timedelta),
    ):
        return True
    return isinstance(value, tuple) and all(_is_immutable_scalar(v) for v in value)


def _columns(value: Mapping[str, ArrayLike]) -> Mapping[str, Array]:
    result: dict[str, Array] = {}
    for name, column in value.items():
        _validate_name(name)
        array = readonly_array(column)
        if _has_complex(array):
            msg = f"Column {name!r} contains complex values; real values are required."
            raise DataValidationError(msg)
        if array.ndim != 1:
            msg = f"Column {name!r} must be one-dimensional; got {array.shape}."
            raise SchemaError(msg)
        result[name] = array
    return MappingProxyType(result)


def _row_ids(value: ArrayLike) -> Array:
    result = readonly_array(value)
    if result.ndim != 1:
        msg = "Row identifiers must be one-dimensional."
        raise SchemaError(msg)
    return result


def _nodes(value: ArrayLike) -> IntArray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        msg = "Node codes must be a one-dimensional integer array."
        raise DataValidationError(msg)
    if np.any(array < 0) or np.any(array > np.iinfo(np.int64).max):
        msg = "Node codes must be nonnegative int64 values."
        raise DataValidationError(msg)
    return cast("IntArray", readonly_array(array, dtype=np.int64))


def _validate_name(name: object) -> None:
    if not isinstance(name, str) or not name.strip():
        msg = "Column names must be nonempty strings."
        raise SchemaError(msg)


def _validate_rows(columns: Mapping[str, Array], row_ids: Array) -> None:
    count = len(row_ids)
    for name, column in columns.items():
        if len(column) != count:
            msg = f"Column {name!r} has {len(column)} rows, expected {count}."
            raise SchemaError(msg)


@attrs.frozen(kw_only=True, eq=False)
class FeatureBatch:
    columns: Mapping[str, Array] = attrs.field(converter=_columns)
    row_ids: Array = attrs.field(converter=_row_ids)

    def __attrs_post_init__(self) -> None:
        _validate_rows(self.columns, self.row_ids)

    @property
    def n_rows(self) -> int:
        return len(self.row_ids)


@attrs.frozen(kw_only=True, eq=False)
class TrainingBatch:
    columns: Mapping[str, Array] = attrs.field(converter=_columns)
    targets: FloatArray = attrs.field(converter=readonly_float_array)
    row_ids: Array = attrs.field(converter=_row_ids)

    def __attrs_post_init__(self) -> None:
        _validate_rows(self.columns, self.row_ids)
        if self.targets.ndim != 1 or len(self.targets) != self.n_rows:
            msg = "Targets must be a vector aligned with the feature rows."
            raise SchemaError(msg)
        if not np.all(np.isfinite(self.targets)):
            msg = "Targets must contain only finite values."
            raise DataValidationError(msg)

    @property
    def n_rows(self) -> int:
        return len(self.row_ids)

    def features(self) -> FeatureBatch:
        return FeatureBatch(columns=self.columns, row_ids=self.row_ids)


@attrs.frozen(kw_only=True, eq=False)
class NumericBatch:
    x: FloatArray = attrs.field(converter=readonly_float_array)
    y: FloatArray = attrs.field(converter=readonly_float_array)
    nodes: IntArray = attrs.field(converter=_nodes)
    row_ids: Array = attrs.field(converter=_row_ids)

    def __attrs_post_init__(self) -> None:
        count = len(self.row_ids)
        if self.x.ndim != 2 or self.x.shape[0] != count:
            msg = "The numerical design matrix must have shape (rows, features)."
            raise SchemaError(msg)
        if self.y.ndim != 1 or len(self.y) != count or len(self.nodes) != count:
            msg = "Targets, node codes, and row identifiers must align with x."
            raise SchemaError(msg)
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.y)):
            msg = "Numerical batches must contain only finite features and targets."
            raise DataValidationError(msg)

    @property
    def n_rows(self) -> int:
        return len(self.row_ids)


class ClosableIterator[T](Protocol):
    def __iter__(self) -> Iterator[T]: ...

    def __next__(self) -> T: ...

    def close(self) -> None: ...


@runtime_checkable
class BatchSource[B](Protocol):
    @property
    def snapshot_id(self) -> str: ...

    @property
    def n_rows(self) -> int | None: ...

    @property
    def column_names(self) -> tuple[str, ...]: ...

    def batches(self) -> ClosableIterator[B]: ...


type FeatureSource = BatchSource[FeatureBatch]
type TrainingSource = BatchSource[TrainingBatch]


def _validate_batch_size(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        msg = "batch_size must be a positive integer."
        raise SchemaError(msg)


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


def _update_array_hash(digest: _Digest, array: Array) -> None:
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    if array.dtype.hasobject:
        encoded = json.dumps(
            [(type(item).__qualname__, repr(item)) for item in array.flat],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(encoded.encode())
    else:
        digest.update(array.tobytes())


def _batch_fingerprint(batch: FeatureBatch | TrainingBatch) -> str:
    digest = hashlib.sha256()
    for name, column in batch.columns.items():
        digest.update(name.encode())
        digest.update(b"\0")
        _update_array_hash(digest, column)
    _update_array_hash(digest, batch.row_ids)
    if isinstance(batch, TrainingBatch):
        _update_array_hash(digest, batch.targets)
    return digest.hexdigest()


@attrs.frozen(kw_only=True, eq=False)
class InMemoryFeatureSource:
    batch: FeatureBatch
    batch_size: int = _DEFAULT_BATCH_SIZE
    _snapshot_id: str = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        _validate_batch_size(self.batch_size)
        object.__setattr__(self, "_snapshot_id", _batch_fingerprint(self.batch))

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def n_rows(self) -> int:
        return self.batch.n_rows

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self.batch.columns)

    def batches(self) -> Generator[FeatureBatch]:
        for start in range(0, self.n_rows, self.batch_size):
            stop = start + self.batch_size
            yield FeatureBatch(
                columns={
                    name: col[start:stop] for name, col in self.batch.columns.items()
                },
                row_ids=self.batch.row_ids[start:stop],
            )


@attrs.frozen(kw_only=True, eq=False)
class InMemoryTrainingSource:
    batch: TrainingBatch
    batch_size: int = _DEFAULT_BATCH_SIZE
    _snapshot_id: str = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        _validate_batch_size(self.batch_size)
        object.__setattr__(self, "_snapshot_id", _batch_fingerprint(self.batch))

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def n_rows(self) -> int:
        return self.batch.n_rows

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self.batch.columns)

    def batches(self) -> Generator[TrainingBatch]:
        for start in range(0, self.n_rows, self.batch_size):
            stop = start + self.batch_size
            yield TrainingBatch(
                columns={
                    name: col[start:stop] for name, col in self.batch.columns.items()
                },
                targets=self.batch.targets[start:stop],
                row_ids=self.batch.row_ids[start:stop],
            )


def _require_module(module: str, extra: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        msg = f"This adapter requires the optional {extra!r} integration."
        raise OptionalDependencyError(msg) from error


def _engine(value: object) -> str:
    for cls in type(value).__mro__:
        candidate = cls.__module__.partition(".")[0]
        if candidate in {"pandas", "polars"}:
            return candidate
    return "numpy"


def _frame_columns(
    x: object, feature_names: Sequence[str] | None
) -> tuple[Mapping[str, Array], Array | None, int | None]:
    engine = _engine(x)
    if engine == "pandas":
        pandas = _require_module("pandas", "pandas")
        if not isinstance(x, pandas.DataFrame):
            msg = "Pandas features must be a DataFrame."
            raise SchemaError(msg)
        names = tuple(x.columns)
        _validate_names(names)
        index = x.index
        if not index.is_unique:
            msg = "Dataframe indexes must be unique for unambiguous row alignment."
            raise SchemaError(msg)
        if feature_names is not None and tuple(feature_names) != names:
            msg = "feature_names must match dataframe column names and order."
            raise SchemaError(msg)
        return (
            {name: x[name].to_numpy() for name in names},
            index.to_numpy(),
            len(index),
        )
    if engine == "polars":
        polars = _require_module("polars", "polars")
        if not isinstance(x, polars.DataFrame):
            msg = "Polars features must be a DataFrame."
            raise SchemaError(msg)
        names = tuple(x.columns)
        _validate_names(names)
        if feature_names is not None and tuple(feature_names) != names:
            msg = "feature_names must match dataframe column names and order."
            raise SchemaError(msg)
        return {name: x[name].to_numpy() for name in names}, None, x.height
    if isinstance(x, Mapping):
        names = tuple(x)
        _validate_names(names)
        if feature_names is not None and tuple(feature_names) != names:
            msg = "feature_names must match mapping keys and order."
            raise SchemaError(msg)
        mapped: dict[str, Array] = {}
        for name, value in x.items():
            assert isinstance(name, str)
            array = np.asarray(value)
            if array.ndim != 1:
                msg = f"Column {name!r} must be one-dimensional; got {array.shape}."
                raise SchemaError(msg)
            mapped[name] = array
        count = len(next(iter(mapped.values()))) if mapped else None
        return mapped, None, count
    try:
        array = np.asarray(x)
    except (TypeError, ValueError) as error:
        msg = "Features must be a mapping, dataframe, or two-dimensional array."
        raise SchemaError(msg) from error
    if array.ndim != 2:
        msg = "Array features must have shape (rows, features)."
        raise SchemaError(msg)
    names = (
        tuple(feature_names)
        if feature_names is not None
        else tuple(f"x{i}" for i in range(array.shape[1]))
    )
    _validate_names(names)
    if len(names) != array.shape[1]:
        msg = "feature_names length must match the array's feature dimension."
        raise SchemaError(msg)
    return dict(zip(names, array.T, strict=True)), None, array.shape[0]


def _validate_names(names: Sequence[object]) -> None:
    for name in names:
        _validate_name(name)
    if len(set(names)) != len(names):
        msg = "Column names must be unique."
        raise SchemaError(msg)


def _targets(x: object, y: object) -> FloatArray:
    if _engine(y) == "pandas":
        pandas = _require_module("pandas", "pandas")
        if not isinstance(y, pandas.Series):
            msg = "Pandas targets must be a Series."
            raise SchemaError(msg)
        index = y.index
        if not index.is_unique:
            msg = "Target indexes must be unique for unambiguous row alignment."
            raise SchemaError(msg)
        if isinstance(x, pandas.DataFrame) and not x.index.equals(index):
            msg = "Feature and target indexes must match exactly and in order."
            raise SchemaError(msg)
        return readonly_float_array(y.to_numpy())
    if _engine(y) == "polars":
        polars = _require_module("polars", "polars")
        if not isinstance(y, polars.Series):
            msg = "Polars targets must be a Series."
            raise SchemaError(msg)
        return readonly_float_array(y.to_numpy())
    return readonly_float_array(np.asarray(y))


def _default_row_ids(
    columns: Mapping[str, Array],
    index: Array | None,
    explicit: ArrayLike | None,
    n_rows: int | None,
) -> Array:
    if explicit is not None:
        result = np.asarray(explicit)
        if result.ndim != 1 or (n_rows is not None and len(result) != n_rows):
            msg = "Row identifiers must be a vector aligned with the feature rows."
            raise SchemaError(msg)
        return result
    if index is not None:
        return index
    if n_rows is not None:
        return np.arange(n_rows, dtype=np.int64)
    if not columns:
        msg = "At least one input column is required to infer the row count."
        raise SchemaError(msg)
    return np.arange(len(np.asarray(next(iter(columns.values())))), dtype=np.int64)


def feature_source_from_data(
    x: object,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    feature_names: Sequence[str] | None = None,
    row_ids: ArrayLike | None = None,
) -> InMemoryFeatureSource:
    columns, index, count = _frame_columns(x, feature_names)
    return InMemoryFeatureSource(
        batch=FeatureBatch(
            columns=columns, row_ids=_default_row_ids(columns, index, row_ids, count)
        ),
        batch_size=batch_size,
    )


def source_from_data(
    x: object,
    y: object,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    feature_names: Sequence[str] | None = None,
    row_ids: ArrayLike | None = None,
) -> InMemoryTrainingSource:
    columns, index, count = _frame_columns(x, feature_names)
    targets = _targets(x, y)
    if count is None and targets.ndim == 1:
        count = len(targets)
    return InMemoryTrainingSource(
        batch=TrainingBatch(
            columns=columns,
            targets=targets,
            row_ids=_default_row_ids(columns, index, row_ids, count),
        ),
        batch_size=batch_size,
    )


@overload
def from_arrays(
    x: ArrayLike,
    y: None = None,
    *,
    feature_names: Sequence[str] | None = None,
    row_ids: ArrayLike | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> InMemoryFeatureSource: ...


@overload
def from_arrays(
    x: ArrayLike,
    y: ArrayLike,
    *,
    feature_names: Sequence[str] | None = None,
    row_ids: ArrayLike | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> InMemoryTrainingSource: ...


def from_arrays(
    x: ArrayLike,
    y: ArrayLike | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    row_ids: ArrayLike | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> InMemoryFeatureSource | InMemoryTrainingSource:
    if y is None:
        return feature_source_from_data(
            x, feature_names=feature_names, row_ids=row_ids, batch_size=batch_size
        )
    return source_from_data(
        x, y, feature_names=feature_names, row_ids=row_ids, batch_size=batch_size
    )


def from_pandas(
    x: object,
    y: object | None = None,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> InMemoryFeatureSource | InMemoryTrainingSource:
    pandas = _require_module("pandas", "pandas")
    if not isinstance(x, pandas.DataFrame):
        msg = "from_pandas requires a pandas DataFrame."
        raise SchemaError(msg)
    if y is None:
        return feature_source_from_data(x, batch_size=batch_size)
    return source_from_data(x, y, batch_size=batch_size)


def from_polars(
    x: object,
    y: object | None = None,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> InMemoryFeatureSource | InMemoryTrainingSource:
    polars = _require_module("polars", "polars")
    if not isinstance(x, polars.DataFrame):
        msg = "from_polars requires a polars DataFrame."
        raise SchemaError(msg)
    if y is None:
        return feature_source_from_data(x, batch_size=batch_size)
    return source_from_data(x, y, batch_size=batch_size)


def _file_token(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as error:
        msg = f"Source file {path} is no longer accessible."
        raise ReplayError(msg) from error
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@attrs.frozen(kw_only=True)
class _ParquetSnapshot:
    path: Path
    feature_columns: tuple[str, ...]
    target: str | None
    n_rows: int
    batch_size: int
    token: tuple[int, int, int, int, int]
    snapshot_id: str

    def verify(self) -> None:
        if _file_token(self.path) != self.token:
            msg = f"Parquet source changed after its snapshot was created: {self.path}."
            raise ReplayError(msg)


def _parquet_batches(
    snapshot: _ParquetSnapshot,
) -> Generator[tuple[Mapping[str, Array], FloatArray | None, IntArray]]:
    snapshot.verify()
    parquet = _require_module("pyarrow.parquet", "parquet")
    reader = parquet.ParquetFile(snapshot.path)
    requested = list(snapshot.feature_columns)
    if snapshot.target is not None:
        requested.append(snapshot.target)
    offset = 0
    try:
        for batch in reader.iter_batches(
            batch_size=snapshot.batch_size, columns=requested
        ):
            snapshot.verify()
            data = batch.to_pydict()
            columns = {
                name: np.asarray(data[name]) for name in snapshot.feature_columns
            }
            targets = (
                None
                if snapshot.target is None
                else readonly_float_array(data[snapshot.target])
            )
            rows = np.arange(offset, offset + batch.num_rows, dtype=np.int64)
            offset += batch.num_rows
            yield columns, targets, rows
        snapshot.verify()
        if offset != snapshot.n_rows:
            msg = "Parquet source row count differs from its snapshot metadata."
            raise ReplayError(msg)
    finally:
        reader.close()


@attrs.frozen(kw_only=True)
class ParquetFeatureSource:
    _snapshot: _ParquetSnapshot = attrs.field(alias="snapshot")

    @property
    def snapshot_id(self) -> str:
        return self._snapshot.snapshot_id

    @property
    def n_rows(self) -> int:
        return self._snapshot.n_rows

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._snapshot.feature_columns

    def batches(self) -> Generator[FeatureBatch]:
        batches = _parquet_batches(self._snapshot)
        try:
            for columns, _, row_ids in batches:
                yield FeatureBatch(columns=columns, row_ids=row_ids)
        finally:
            batches.close()


@attrs.frozen(kw_only=True)
class ParquetTrainingSource:
    _snapshot: _ParquetSnapshot = attrs.field(alias="snapshot")

    @property
    def snapshot_id(self) -> str:
        return self._snapshot.snapshot_id

    @property
    def n_rows(self) -> int:
        return self._snapshot.n_rows

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._snapshot.feature_columns

    def batches(self) -> Generator[TrainingBatch]:
        batches = _parquet_batches(self._snapshot)
        try:
            for columns, targets, row_ids in batches:
                assert targets is not None
                yield TrainingBatch(columns=columns, targets=targets, row_ids=row_ids)
        finally:
            batches.close()


@overload
def parquet_source(
    path: str | Path,
    *,
    target: None = None,
    columns: Sequence[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> ParquetFeatureSource: ...


@overload
def parquet_source(
    path: str | Path,
    *,
    target: str,
    columns: Sequence[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> ParquetTrainingSource: ...


def parquet_source(
    path: str | Path,
    *,
    target: str | None = None,
    columns: Sequence[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> ParquetFeatureSource | ParquetTrainingSource:
    _validate_batch_size(batch_size)
    parquet = _require_module("pyarrow.parquet", "parquet")
    resolved = Path(path).resolve(strict=True)
    token = _file_token(resolved)
    reader = parquet.ParquetFile(resolved)
    try:
        names = tuple(reader.schema_arrow.names)
        _validate_names(names)
        if target is not None and target not in names:
            msg = f"Target column {target!r} does not exist in the Parquet source."
            raise SchemaError(msg)
        selected = (
            tuple(columns)
            if columns is not None
            else tuple(n for n in names if n != target)
        )
        _validate_names(selected)
        if any(name not in names for name in selected):
            msg = "Requested feature columns do not exist in the Parquet source."
            raise SchemaError(msg)
        if target is not None and target in selected:
            msg = "The target column cannot also be selected as a feature."
            raise SchemaError(msg)
        n_rows = reader.metadata.num_rows
    finally:
        reader.close()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if _file_token(resolved) != token:
        msg = "Parquet source changed while its snapshot was being created."
        raise ReplayError(msg)
    digest.update(json.dumps((selected, target)).encode())
    snapshot = _ParquetSnapshot(
        path=resolved,
        feature_columns=selected,
        target=target,
        n_rows=n_rows,
        batch_size=batch_size,
        token=token,
        snapshot_id=digest.hexdigest(),
    )
    if target is None:
        return ParquetFeatureSource(snapshot=snapshot)
    return ParquetTrainingSource(snapshot=snapshot)
