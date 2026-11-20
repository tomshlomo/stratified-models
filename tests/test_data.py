from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from stratified_models.data import (
    FeatureBatch,
    NumericBatch,
    TrainingBatch,
    feature_source_from_data,
    from_arrays,
    from_pandas,
    from_polars,
    parquet_source,
    source_from_data,
)
from stratified_models.errors import DataValidationError, ReplayError, SchemaError


def test_batches_own_readonly_input_data_and_column_mapping() -> None:
    values = np.array([1.0, 2.0])
    row_ids = np.array([7, 8])
    columns = {"a": values}
    batch = FeatureBatch(columns=columns, row_ids=row_ids)
    values[0] = 100.0
    row_ids[0] = 100
    columns["new"] = values
    assert batch.columns["a"].tolist() == [1.0, 2.0]
    assert batch.row_ids.tolist() == [7, 8]
    assert tuple(batch.columns) == ("a",)
    with pytest.raises(ValueError, match="read-only"):
        batch.columns["a"][0] = 0
    with pytest.raises(ValueError):
        batch.columns["a"].setflags(write=True)
    with pytest.raises(TypeError):
        batch.columns["other"] = np.array([1, 2])


def test_heterogeneous_immutable_object_columns_preserve_label_types() -> None:
    labels = np.array([1, "1"], dtype=object)
    batch = FeatureBatch(columns={"group": labels}, row_ids=[0, 1])
    labels[0] = 9
    assert batch.columns["group"].tolist() == [1, "1"]
    with pytest.raises(DataValidationError, match="immutable scalar"):
        FeatureBatch(columns={"bad": np.array([{}], dtype=object)}, row_ids=[0])


@pytest.mark.parametrize(
    "columns,row_ids",
    [({"a": [[1, 2]]}, [0]), ({"a": [1]}, [0, 1]), ({"": [1]}, [0])],
)
def test_feature_schema_rejects_invalid_shapes_and_names(
    columns: dict[str, Sequence[object]], row_ids: Sequence[int]
) -> None:
    with pytest.raises(SchemaError):
        FeatureBatch(columns=columns, row_ids=row_ids)


def test_targets_and_numerical_batches_reject_ambiguous_or_nonfinite_values() -> None:
    with pytest.raises(SchemaError, match="vector"):
        TrainingBatch(columns={"a": [1]}, targets=[[1]], row_ids=[0])
    with pytest.raises(DataValidationError, match="finite"):
        TrainingBatch(columns={"a": [1]}, targets=[np.nan], row_ids=[0])
    with pytest.raises(DataValidationError, match="finite"):
        NumericBatch(x=[[np.inf]], y=[0], nodes=[0], row_ids=[0])
    with pytest.raises(DataValidationError, match="integer"):
        NumericBatch(x=[[1]], y=[0], nodes=[0.5], row_ids=[0])
    with pytest.raises(DataValidationError, match="nonnegative"):
        NumericBatch(x=[[1]], y=[0], nodes=[-1], row_ids=[0])
    with pytest.raises(DataValidationError, match="real numerical"):
        NumericBatch(x=np.array([[1 + 2j]]), y=[0], nodes=[0], row_ids=[0])
    batch = NumericBatch(x=[[1]], y=[0], nodes=[0], row_ids=[3])
    assert batch.nodes.dtype == np.int64
    assert batch.x.dtype == np.float64
    assert not batch.x.flags.writeable


def test_inmemory_source_is_replayable_owned_and_partition_independent() -> None:
    values = np.arange(7.0)
    targets = values * 2
    source = source_from_data({"a": values}, targets, batch_size=3)
    other_partition = source_from_data({"a": values}, targets, batch_size=2)
    assert source.snapshot_id == other_partition.snapshot_id
    assert source.n_rows == 7
    assert source.column_names == ("a",)
    values[:] = 100
    targets[:] = 0
    first_pass = list(source.batches())
    second_pass = list(source.batches())
    assert [b.n_rows for b in first_pass] == [3, 3, 1]
    assert np.array_equal(
        np.concatenate([b.targets for b in second_pass]), np.arange(7) * 2
    )
    assert np.array_equal(np.concatenate([b.row_ids for b in first_pass]), np.arange(7))
    assert source.snapshot_id != source_from_data({"a": values}, targets).snapshot_id
    iterator = source.batches()
    next(iterator)
    iterator.close()
    assert len(list(source.batches())) == 3


def test_array_adapter_validates_names_and_retains_positions() -> None:
    source = from_arrays([[1, 2], [3, 4]], [5, 6], feature_names=("a", "b"))
    batch = next(source.batches())
    assert batch.columns["b"].tolist() == [2, 4]
    assert batch.targets.tolist() == [5, 6]
    assert batch.row_ids.tolist() == [0, 1]
    assert from_arrays([[1, 2]]).column_names == ("x0", "x1")
    with pytest.raises(SchemaError, match="unique"):
        from_arrays([[1, 2]], feature_names=("a", "a"))
    with pytest.raises(SchemaError, match="dimension"):
        from_arrays([[1, 2]], feature_names=("a",))
    with pytest.raises(SchemaError, match="positive integer"):
        from_arrays([[1]], batch_size=0)
    with pytest.raises(SchemaError, match="one-dimensional"):
        source_from_data({"a": 1}, [1])


def test_empty_arrays_have_known_row_count_and_replay_empty() -> None:
    source = from_arrays(np.empty((0, 2)), np.empty(0))
    assert source.n_rows == 0
    assert list(source.batches()) == []


def test_zero_column_arrays_allow_explicit_intercept_only_models() -> None:
    source = from_arrays(np.empty((3, 0)), np.array([1, 2, 3]))
    assert source.n_rows == 3
    assert source.column_names == ()
    assert next(source.batches()).row_ids.tolist() == [0, 1, 2]
    features = feature_source_from_data(np.empty((3, 0)))
    assert features.n_rows == 3
    assert next(features.batches()).columns == {}
    assert source_from_data({}, [1, 2]).n_rows == 2
    with pytest.raises(SchemaError, match="aligned"):
        from_arrays(np.empty((3, 0)), [1, 2], row_ids=[0, 1])


def test_pandas_adapter_owns_data_preserves_rows_and_rejects_ambiguous_indexes() -> (
    None
):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=[10, 11])
    target = pd.Series([3.0, 4.0], index=[10, 11])
    source = from_pandas(frame, target)
    frame.loc[10, "a"] = 100
    target.loc[10] = 100
    batch = next(source.batches())
    assert batch.columns["a"].tolist() == [1.0, 2.0]
    assert batch.targets.tolist() == [3.0, 4.0]
    assert batch.row_ids.tolist() == [10, 11]
    with pytest.raises(SchemaError, match="exactly"):
        source_from_data(frame, target.iloc[::-1])
    duplicate_index = pd.DataFrame({"a": [1, 2]}, index=[0, 0])
    with pytest.raises(SchemaError, match="unique"):
        feature_source_from_data(duplicate_index)
    duplicate_columns = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(SchemaError, match="unique"):
        from_pandas(duplicate_columns)


def test_polars_adapter_preserves_types_without_mutating_frame() -> None:
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"a": [1.0, 2.0], "group": ["n", "s"]})
    source = from_polars(frame, pl.Series([3, 4]), batch_size=1)
    batches = list(source.batches())
    assert source.column_names == ("a", "group")
    assert batches[1].columns["group"].tolist() == ["s"]
    assert batches[0].targets.tolist() == [3.0]
    assert frame.columns == ["a", "group"]


def test_parquet_replays_partitions_and_rejects_file_changes(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"a": range(7), "y": np.arange(7) * 2}), path)
    source = parquet_source(path, target="y", batch_size=3)
    assert source.n_rows == 7
    assert source.column_names == ("a",)
    assert [b.n_rows for b in source.batches()] == [3, 3, 1]
    assert (
        np.concatenate([b.targets for b in source.batches()]).tolist()
        == (np.arange(7) * 2).tolist()
    )
    iterator = source.batches()
    next(iterator)
    iterator.close()
    assert len(list(source.batches())) == 3
    pq.write_table(pa.table({"a": range(8), "y": range(8)}), path)
    with pytest.raises(ReplayError, match="changed"):
        next(source.batches())


def test_parquet_feature_only_and_target_selection(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"a": [1], "b": [2], "y": [3]}), path)
    source = parquet_source(path, columns=("a",))
    assert next(source.batches()).columns["a"].tolist() == [1]
    with pytest.raises(SchemaError, match="also"):
        parquet_source(path, target="y", columns=("a", "y"))
    with pytest.raises(SchemaError, match="does not exist"):
        parquet_source(path, target="missing")


def test_parquet_closes_reader_when_iteration_stops_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"a": range(5), "y": range(5)}), path)
    original_close = pq.ParquetFile.close
    closed: list[object] = []

    def recording_close(reader: object, *, force: bool = False) -> None:
        closed.append(reader)
        original_close(reader, force=force)

    monkeypatch.setattr(pq.ParquetFile, "close", recording_close)
    source = parquet_source(path, target="y", batch_size=2)
    assert len(closed) == 1  # Metadata reader.
    iterator = source.batches()
    next(iterator)
    assert len(closed) == 1
    iterator.close()
    assert len(closed) == 2


@pytest.mark.parametrize("dtype", [np.complex128, object])
def test_complex_features_and_targets_fail_before_discarding_imaginary_parts(
    dtype: object,
) -> None:
    complex_values = np.array(
        [np.complex128(1 + 2j), np.complex128(3 - 1j)], dtype=dtype
    )
    with pytest.raises(DataValidationError, match="real"):
        source_from_data({"x": complex_values}, [0.0, 1.0])
    with pytest.raises(DataValidationError, match="real"):
        source_from_data({"x": [0.0, 1.0]}, complex_values)
