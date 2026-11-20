"""Out-of-core preparation and optimization do not retain sample history."""

from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np

from stratified_models.data import TrainingBatch
from stratified_models.fit import FitSuccess, fit_batches
from stratified_models.prepared import PreparationOptions
from stratified_models.preprocessing import AddIntercept, Standardize, compose
from stratified_models.solvers import (
    ConjugateGradient,
    SolverCache,
)
from stratified_models.specs import L2, Axis, Graph, Laplacian, Pipeline, RegressionSpec


@dataclass
class _GeneratedSource:
    count: int
    batch_size: int = 128
    active_iterators: int = 0
    max_active_iterators: int = 0
    largest_batch: int = 0

    @property
    def snapshot_id(self) -> str:
        return f"generated-v1-{self.count}"

    @property
    def n_rows(self) -> int:
        return self.count

    @property
    def column_names(self) -> tuple[str, ...]:
        return ("x", "region")

    def batches(self) -> Generator[TrainingBatch]:
        self.active_iterators += 1
        self.max_active_iterators = max(
            self.max_active_iterators, self.active_iterators
        )
        try:
            for first in range(0, self.count, self.batch_size):
                ids = np.arange(first, min(first + self.batch_size, self.count))
                x = (ids % 37 - 18) / 10
                region = ids % 2
                self.largest_batch = max(self.largest_batch, len(ids))
                yield TrainingBatch(
                    columns={"x": x, "region": region},
                    targets=1 + 0.7 * x + 0.2 * region,
                    row_ids=ids,
                )
        finally:
            self.active_iterators -= 1


def _profile(count: int) -> int:
    axis = Axis(name="region", nodes=(0, 1))
    pipeline = Pipeline(
        preprocessing=compose(
            Standardize(columns=("x",)), AddIntercept(name="intercept")
        ),
        model=RegressionSpec(
            features=("intercept", "x"),
            axes=(axis,),
            penalties=(
                L2(name="ridge", weight=0.01, features=("x",)),
                Laplacian(
                    name="space",
                    weight=0.1,
                    features=("intercept", "x"),
                    graph=Graph.path(axis=axis),
                ),
            ),
        ),
    )
    source = _GeneratedSource(count)
    gc.collect()
    tracemalloc.start()
    try:
        result = fit_batches(
            pipeline,
            source,
            solver=ConjugateGradient(),
            options=PreparationOptions(statistics_max_bytes=0),
            cache=SolverCache(max_bytes=0),
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert isinstance(result, FitSuccess), result
    assert result.diagnostics.source_passes > 0
    assert source.active_iterators == 0
    assert source.max_active_iterators == 1
    assert source.largest_batch <= 128
    return peak


def test_generated_sample_memory_stays_bounded_as_sample_count_grows_tenfold() -> None:
    # Compile the fixed batch/schema kernels before comparing allocation peaks.
    _profile(128)
    small = _profile(1_024)
    large = _profile(10_240)
    # Fixed batch/state memory, allowing runtime bookkeeping and tracing noise.
    assert large < 3 * small + 1_000_000, (small, large)
