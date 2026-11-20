from collections.abc import Generator, Sequence

import numpy as np
import pytest

from stratified_models.data import TrainingBatch, source_from_data
from stratified_models.errors import (
    DataValidationError,
    ReplayError,
    SchemaError,
    ValidationError,
)
from stratified_models.preprocessing import (
    AddIntercept,
    FittedStandardize,
    FittedTransformChain,
    FixedWidthBins,
    Standardize,
    TransformChain,
    compose,
)


def test_streaming_standardization_matches_population_statistics() -> None:
    x = np.array([1.0, 4.0, 2.0, 10.0, 8.0]) + 1e9
    source = source_from_data(
        {"x": x, "constant": np.full(5, 7.0)}, np.arange(5), batch_size=2
    )
    fitted = Standardize(columns=("x", "constant")).fit(source)
    np.testing.assert_allclose(fitted.means, [x.mean(), 7.0])
    np.testing.assert_allclose(fitted.scales, [x.std(ddof=0), 1.0])
    transformed = fitted.transform(source.batch)
    np.testing.assert_allclose(
        transformed.columns["x"], (x - x.mean()) / x.std(ddof=0), atol=1e-7
    )
    np.testing.assert_array_equal(transformed.columns["constant"], np.zeros(5))
    np.testing.assert_array_equal(source.batch.columns["x"], x)
    np.testing.assert_array_equal(transformed.targets, source.batch.targets)
    np.testing.assert_array_equal(transformed.row_ids, source.batch.row_ids)


def test_transform_chain_preserves_inputs_and_replays_deterministically() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    source = source_from_data({"x": x}, x, row_ids=np.arange(10, 15), batch_size=2)
    chain = compose(
        FixedWidthBins(column="x", output="bin", origin=0, width=2, n_bins=3),
        Standardize(columns=("x",)),
        AddIntercept(name="intercept"),
    )
    fitted = chain.fit(source)
    wrapped = fitted.wrap_source(source)
    assert wrapped.column_names == ("x", "bin", "intercept")
    assert wrapped.n_rows == 5
    assert wrapped.snapshot_id == fitted.wrap_source(source).snapshot_id
    first = list(wrapped.batches())
    second = list(wrapped.batches())
    np.testing.assert_array_equal(
        np.concatenate([batch.columns["bin"] for batch in first]), [0, 0, 1, 1, 2]
    )
    for before, after in zip(first, second, strict=True):
        np.testing.assert_array_equal(before.row_ids, after.row_ids)
        np.testing.assert_array_equal(before.columns["x"], after.columns["x"])
        np.testing.assert_array_equal(before.targets, after.targets)
    assert tuple(source.batch.columns) == ("x",)
    np.testing.assert_array_equal(x, source.batch.columns["x"])


def test_bin_grid_has_explicit_half_open_bounds_and_rejects_collisions() -> None:
    recipe = FixedWidthBins(column="x", output="bin", origin=-2, width=2, n_bins=2)
    source = source_from_data({"x": [-2.0, 0.0, np.nextafter(2.0, -np.inf)]}, [0, 0, 0])
    fitted = recipe.fit(source)
    np.testing.assert_array_equal(
        fitted.transform(source.batch).columns["bin"], [0, 1, 1]
    )
    for value in (-2.01, 2.0, np.nan):
        invalid = source_from_data({"x": [value]}, [0])
        with pytest.raises(DataValidationError):
            fitted.transform(invalid.batch)
    with pytest.raises(SchemaError, match="already exists"):
        AddIntercept(name="x").fit(source)
    with pytest.raises(SchemaError, match="missing"):
        Standardize(columns=("missing",)).fit(source)


def test_fitted_parameters_own_immutable_buffers_and_can_be_prefitted() -> None:
    means = np.array([3.0])
    scales = np.array([2.0])
    fitted = FittedStandardize(columns=("x",), means=means, scales=scales)
    means[0] = 99
    scales[0] = 99
    assert fitted.means[0] == 3
    assert fitted.scales[0] == 2
    with pytest.raises(ValueError):
        fitted.means[0] = 9
    with pytest.raises(ValueError):
        fitted.scales.setflags(write=True)
    source = source_from_data({"x": [3.0, 5.0]}, [0, 1])
    chain = TransformChain(steps=(fitted,)).fit(source)
    np.testing.assert_array_equal(
        chain.transform(source.batch).columns["x"], [0.0, 1.0]
    )


def test_unknown_row_count_is_supported_and_declared_counts_are_checked() -> None:
    source = source_from_data({"x": [1.0, 2.0, 3.0]}, [0, 1, 2], batch_size=2)

    class Source:
        snapshot_id = "stable"
        column_names = ("x",)

        def __init__(self, count: int | None) -> None:
            self.n_rows = count

        def batches(self) -> Generator[TrainingBatch]:
            return source.batches()

    fitted = Standardize(columns=("x",)).fit(Source(None))
    assert fitted.means[0] == 2.0
    with pytest.raises(ReplayError, match="declared 4"):
        Standardize(columns=("x",)).fit(Source(4))


def test_custom_fitted_transform_cannot_reorder_rows_or_targets() -> None:
    source = source_from_data({"x": [1.0, 2.0]}, [0.0, 1.0])

    class Reorder:
        fingerprint = "reorder"

        def output_columns(self, columns: Sequence[str]) -> tuple[str, ...]:
            return tuple(columns)

        def transform(self, batch: TrainingBatch) -> TrainingBatch:
            return TrainingBatch(
                columns={"x": batch.columns["x"][::-1]},
                targets=batch.targets[::-1],
                row_ids=batch.row_ids[::-1],
            )

    with pytest.raises(SchemaError, match="row identities"):
        FittedTransformChain(steps=(Reorder(),)).transform(source.batch)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"width": -1},
        {"width": np.inf},
        {"n_bins": 0},
        {"n_bins": True},
        {"origin": np.nan},
    ],
)
def test_invalid_bin_grids_fail_before_fitting(kwargs: dict) -> None:
    params = {
        "column": "x",
        "output": "bin",
        "origin": 0,
        "width": 1,
        "n_bins": 2,
        **kwargs,
    }
    with pytest.raises(ValidationError):
        FixedWidthBins(**params)
