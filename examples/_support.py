"""Presentation and deterministic experiment helpers shared by the notebooks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import TYPE_CHECKING

import attrs
import numpy as np
import pandas as pd

from stratified_models.data import FeatureBatch, TrainingBatch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from stratified_models import FitResult
    from stratified_models.solvers import Diagnostics

COLORS: Mapping[str, str] = MappingProxyType(
    {
        "truth": "#1f2937",
        "pooled": "#9a6a36",
        "independent": "#7c6ab0",
        "stratified": "#087f8c",
        "ADMM": "#087f8c",
        "Direct": "#b56028",
        "CG": "#6554a4",
        "Newton-CG": "#4477aa",
        "CVXPY": "#aa3377",
        "reference": "#606c76",
        "accent": "#d49128",
        "train": "#4477aa",
        "test": "#087f8c",
    }
)

type _Batch = FeatureBatch | TrainingBatch


@attrs.frozen(kw_only=True)
class OffsetFeature:
    """A tiny external fitted transform for the custom-component walkthrough."""

    column: str
    offset: float

    def __attrs_post_init__(self) -> None:
        if not self.column or not np.isfinite(self.offset):
            raise ValueError("OffsetFeature requires a column and finite offset.")

    @property
    def fingerprint(self) -> str:
        state = repr(("demo-offset-v1", self.column, self.offset)).encode()
        return hashlib.sha256(state).hexdigest()

    def output_columns(self, columns: Sequence[str]) -> tuple[str, ...]:
        if self.column not in columns:
            raise ValueError(f"Missing offset column {self.column!r}.")
        return tuple(columns)

    def transform[B: _Batch](self, batch: B) -> B:
        self.output_columns(tuple(batch.columns))
        columns = {
            **batch.columns,
            self.column: np.asarray(batch.columns[self.column], dtype=float)
            + self.offset,
        }
        return attrs.evolve(batch, columns=columns)


def style_axes(
    ax: Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Apply the notebooks' quiet plotting style to one axes object."""
    ax.set_facecolor("#fbfcfd")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d4dce3")
    ax.tick_params(colors="#465362", labelsize=9)
    ax.grid(axis="y", alpha=0.18, color="#64748b")
    ax.set_axisbelow(True)
    if title is not None:
        ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color="#465362", fontsize=10)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color="#465362", fontsize=10)


@contextmanager
def plot_style() -> Iterator[None]:
    """Scope matplotlib settings rather than changing a process-wide style."""
    import matplotlib as mpl

    with mpl.rc_context(
        {
            "figure.facecolor": "white",
            "figure.dpi": 120,
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
        }
    ):
        yield


def require_success[T](result: FitResult[T]) -> T:
    """Make a failed demonstration visible instead of plotting an invalid fit."""
    from stratified_models import FitFailure

    if isinstance(result, FitFailure):
        message = (
            f"Fit failed: {result.reason}; termination={result.diagnostics.termination}"
        )
        raise RuntimeError(message)
    return result.predictor


def diagnostics_summary(info: Diagnostics) -> dict[str, object]:
    """Return display-ready public diagnostics for one completed solve."""
    return {
        "solver": info.solver,
        "converged": info.converged,
        "termination": info.termination,
        "objective": None if info.objective is None else info.objective.total,
        "stationarity": info.stationarity,
        "iterations": info.iterations,
        "solve seconds": info.elapsed_seconds,
        "solve passes": info.source_passes,
        "solve rows": info.rows_scanned,
        "cache hits": info.cache.hits,
        "factorizations": info.cache.factorizations,
    }


def fit_cost_summary[T](result: FitResult[T]) -> dict[str, object]:
    """Keep one-off preparation work separate from the numerical solve."""
    preparation = result.preparation
    if preparation is None:
        return {"prepare passes": None, "prepare rows": None, "prepare seconds": None}
    return {
        "prepare passes": preparation.source_passes,
        "prepare rows": preparation.rows_scanned,
        "prepare seconds": preparation.elapsed_seconds,
        "statistics KiB": preparation.statistics_bytes / 1024,
        "total passes": preparation.source_passes + result.diagnostics.source_passes,
        "prepare + solve seconds": preparation.elapsed_seconds
        + result.diagnostics.elapsed_seconds,
    }


def regression_metrics(y_true: ArrayLike, prediction: ArrayLike) -> dict[str, float]:
    """Regression metrics with an explicit population mean convention."""
    truth = np.asarray(y_true, dtype=float)
    fitted = np.asarray(prediction, dtype=float)
    squared = (truth - fitted) ** 2
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(squared))),
        "MAE": float(np.mean(np.abs(truth - fitted))),
        "R²": 1.0 - float(np.sum(squared)) / denominator
        if denominator
        else float("nan"),
    }


def classification_metrics(
    y_true: ArrayLike, probabilities: ArrayLike
) -> dict[str, float]:
    """Proper probability scores and a fixed-threshold classification error."""
    truth = np.asarray(y_true, dtype=float)
    positive = np.asarray(probabilities, dtype=float)
    if positive.ndim == 2:
        positive = positive[:, 1]
    safe = np.clip(positive, 1e-12, 1.0 - 1e-12)
    return {
        "Log loss": float(
            -np.mean(truth * np.log(safe) + (1 - truth) * np.log1p(-safe))
        ),
        "Brier score": float(np.mean((truth - positive) ** 2)),
        "Error @ 0.5": float(np.mean((positive >= 0.5) != truth)),
    }


def synthetic_regression(
    *,
    seed: int = 17,
    n_groups: int = 8,
    train_per_group: int = 24,
    test_per_group: int = 80,
    missing_group: int | None = 4,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, NDArray[np.float64]]:
    """A fixed smooth coefficient field with one genuinely irrelevant feature."""
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 2.0 * np.pi, n_groups, endpoint=False)
    truth = np.column_stack(
        (1.2 + 0.7 * np.sin(phase), np.zeros(n_groups), 0.6 * np.cos(phase))
    )
    train_frames = []
    test_frames = []
    train_targets = []
    test_targets = []
    for group in range(n_groups):
        train_rows = 0 if group == missing_group else train_per_group
        for count, frames, targets in (
            (train_rows, train_frames, train_targets),
            (test_per_group, test_frames, test_targets),
        ):
            x = rng.normal(size=(count, 2))
            y = (
                x @ truth[group, :2]
                + truth[group, 2]
                + rng.normal(scale=0.7, size=count)
            )
            frames.append(
                pd.DataFrame({"x": x[:, 0], "noise": x[:, 1], "group": group})
            )
            targets.append(y)
    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)
    y_train = pd.Series(np.concatenate(train_targets), index=train.index, name="target")
    y_test = pd.Series(np.concatenate(test_targets), index=test.index, name="target")
    return train, y_train, test, y_test, truth


def synthetic_classification(
    *,
    seed: int = 23,
    n_groups: int = 5,
    n_train: int = 360,
    n_test: int = 800,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Generate each complete snapshot before dividing it into any batches."""
    rng = np.random.default_rng(seed)

    def draw(count: int) -> tuple[pd.DataFrame, pd.Series]:
        group = rng.integers(0, n_groups, size=count)
        x = rng.normal(size=(count, 2))
        beta = 0.7 + 0.45 * np.sin(group * np.pi / max(1, n_groups - 1))
        logit = beta * x[:, 0] - 0.6 * x[:, 1] + 0.5 * np.cos(group)
        probability = 1.0 / (1.0 + np.exp(-logit))
        frame = pd.DataFrame({"x": x[:, 0], "noise": x[:, 1], "group": group})
        target = pd.Series(
            rng.binomial(1, probability), index=frame.index, name="target"
        )
        return frame, target

    train, y_train = draw(n_train)
    test, y_test = draw(n_test)
    return train, y_train, test, y_test


def calibration_bins(
    y_true: ArrayLike, probabilities: ArrayLike, *, bins: int = 8
) -> pd.DataFrame:
    """Equal-width calibration bins with counts, omitting empty bins."""
    truth = np.asarray(y_true, dtype=float)
    positive = np.asarray(probabilities, dtype=float)
    if positive.ndim == 2:
        positive = positive[:, 1]
    index = np.minimum((positive * bins).astype(int), bins - 1)
    rows = []
    for bin_index in range(bins):
        mask = index == bin_index
        if np.any(mask):
            rows.append(
                {
                    "mean probability": float(np.mean(positive[mask])),
                    "observed fraction": float(np.mean(truth[mask])),
                    "count": int(np.sum(mask)),
                }
            )
    return pd.DataFrame(rows)


def package_versions(
    packages: Sequence[str] = ("stratified-models", "jax", "numpy", "marimo"),
) -> dict[str, str]:
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "source checkout"
    return versions
