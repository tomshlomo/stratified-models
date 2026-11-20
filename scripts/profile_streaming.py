"""Measure fresh-process peak RSS for a generated replayable sample stream.

Examples::

    uv run python scripts/profile_streaming.py --samples 20000 --prepare-only
    uv run python scripts/profile_streaming.py --samples 2000000 --prepare-only
    uv run python scripts/profile_streaming.py --samples 20000 --fit

The fixed schema contains eight numerical features and four strata. Values are
functions of row identifiers, so batch size changes neither data nor ordering.
No sample-sized array or collection is constructed. RSS includes the interpreter
and imported libraries; compare separate invocations rather than reusing a process.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Generator

import attrs
import numpy as np

from stratified_models.data import TrainingBatch
from stratified_models.fit import FitFailure, SolveSession
from stratified_models.prepared import PreparationOptions, prepare_batches
from stratified_models.preprocessing import AddIntercept, Standardize, compose
from stratified_models.solvers import ConjugateGradient, SolverCache
from stratified_models.specs import L2, Axis, Graph, Laplacian, Pipeline, RegressionSpec

_FEATURE_NAMES = tuple(f"x{index}" for index in range(8))
_NODE_COUNT = 4


@attrs.define(kw_only=True)
class GeneratedSource:
    """Instrumented deterministic source with memory proportional to batch size."""

    samples: int
    batch_size: int
    passes: int = attrs.field(init=False, default=0)
    rows_scanned: int = attrs.field(init=False, default=0)
    active_iterators: int = attrs.field(init=False, default=0)
    max_active_iterators: int = attrs.field(init=False, default=0)
    largest_batch: int = attrs.field(init=False, default=0)

    def __attrs_post_init__(self) -> None:
        if self.samples <= 0 or self.batch_size <= 0:
            message = "samples and batch_size must be positive"
            raise ValueError(message)

    @property
    def snapshot_id(self) -> str:
        return f"profile-stream-v1:{self.samples}"

    @property
    def n_rows(self) -> int:
        return self.samples

    @property
    def column_names(self) -> tuple[str, ...]:
        return (*_FEATURE_NAMES, "region")

    def batches(self) -> Generator[TrainingBatch]:
        self.passes += 1
        self.active_iterators += 1
        self.max_active_iterators = max(
            self.max_active_iterators, self.active_iterators
        )
        try:
            for start in range(0, self.samples, self.batch_size):
                ids = np.arange(
                    start, min(start + self.batch_size, self.samples), dtype=np.int64
                )
                columns = {
                    name: np.sin(ids * (index + 1) * 0.071)
                    + ((ids + index * 7) % 31 - 15) / 20
                    for index, name in enumerate(_FEATURE_NAMES)
                }
                region = ids % _NODE_COUNT
                targets = 0.4 + 0.1 * region + 0.01 * np.cos(ids * 0.013)
                for index, name in enumerate(_FEATURE_NAMES):
                    targets = targets + (index + 1) / 16 * columns[name]
                self.rows_scanned += len(ids)
                self.largest_batch = max(self.largest_batch, len(ids))
                yield TrainingBatch(
                    columns={**columns, "region": region}, targets=targets, row_ids=ids
                )
        finally:
            self.active_iterators -= 1


def _recipe() -> Pipeline[RegressionSpec]:
    axis = Axis(name="region", nodes=tuple(range(_NODE_COUNT)))
    features = (*_FEATURE_NAMES, "intercept")
    return Pipeline(
        preprocessing=compose(
            Standardize(columns=_FEATURE_NAMES), AddIntercept(name="intercept")
        ),
        model=RegressionSpec(
            features=features,
            axes=(axis,),
            penalties=(
                L2(name="ridge", weight=0.01, features=_FEATURE_NAMES),
                Laplacian(
                    name="smooth",
                    weight=0.1,
                    graph=Graph.path(axis=axis),
                    features=features,
                ),
            ),
        ),
    )


def _peak_rss_bytes() -> int:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _positive(value: str) -> int:
    count = int(value)
    if count <= 0:
        message = "must be a positive integer"
        raise argparse.ArgumentTypeError(message)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=_positive, default=20_000)
    parser.add_argument("--batch-size", type=_positive, default=4_096)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--prepare-only", dest="mode", action="store_const", const="prepare"
    )
    modes.add_argument("--fit", dest="mode", action="store_const", const="fit")
    parser.set_defaults(mode="fit")
    args = parser.parse_args()
    if sys.platform == "win32":
        parser.error("this RSS profiler requires the Unix resource module")

    source = GeneratedSource(samples=args.samples, batch_size=args.batch_size)
    baseline = _peak_rss_bytes()
    started = time.perf_counter()
    prepared = prepare_batches(
        _recipe(),
        source,
        options=PreparationOptions(batch_size=args.batch_size, statistics_max_bytes=0),
    )
    preparation = prepared.preparation
    fit_report: dict[str, object] | None = None
    exit_code = 0
    if args.mode == "fit":
        outcome = SolveSession(
            prepared,
            solver=ConjugateGradient(tol=1e-8, max_iterations=100, max_passes=150),
            cache=SolverCache(max_bytes=0),
        ).solve()
        diagnostics = outcome.diagnostics
        fit_report = {
            "converged": diagnostics.converged,
            "solver": diagnostics.solver,
            "termination": diagnostics.termination,
            "iterations": diagnostics.iterations,
            "source_passes": diagnostics.source_passes,
            "rows_scanned": diagnostics.rows_scanned,
            "elapsed_seconds": diagnostics.elapsed_seconds,
            "objective": None
            if diagnostics.objective is None
            else diagnostics.objective.total,
            "stationarity": diagnostics.stationarity,
        }
        if isinstance(outcome, FitFailure):
            fit_report["failure"] = {
                "code": outcome.reason.code,
                "message": outcome.reason.message,
            }
            exit_code = 1
    peak = _peak_rss_bytes()
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "samples": source.samples,
        "batch_size": source.batch_size,
        "snapshot_id": source.snapshot_id,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak,
        "import_peak_rss_bytes": baseline,
        "processing_peak_increase_bytes": max(0, peak - baseline),
        "source_passes": source.passes,
        "rows_scanned": source.rows_scanned,
        "largest_batch": source.largest_batch,
        "max_active_iterators": source.max_active_iterators,
        "active_iterators": source.active_iterators,
        "model": {
            "nodes": prepared.n_nodes,
            "features": prepared.n_features,
            "coefficients": prepared.n_nodes * prepared.n_features,
            "coefficient_bytes": prepared.n_nodes
            * prepared.n_features
            * np.dtype(np.float64).itemsize,
        },
        "preparation": {
            "source_passes": preparation.source_passes,
            "rows_scanned": preparation.rows_scanned,
            "elapsed_seconds": preparation.elapsed_seconds,
            "statistics_bytes": preparation.statistics_bytes,
        },
        "fit": fit_report,
    }
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
