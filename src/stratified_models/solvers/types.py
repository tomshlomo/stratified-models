"""Immutable configuration, problem contracts, and numerical results."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from stratified_models.data import NumericBatch
    from stratified_models.specs import L1, L2, Axis, Laplacian

FloatArray = NDArray[np.float64]


def _owned_array(value: FloatArray) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        msg = "Numerical state must contain only finite values."
        raise ValueError(msg)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        msg = f"{name} must be finite and positive."
        raise ValueError(msg)


def _count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer."
        raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class QuadraticStatistics:
    """Unnormalized per-node cross products; normalize by global sample count."""

    xtx: FloatArray
    xty: FloatArray
    yty: float

    def __post_init__(self) -> None:
        xtx = _owned_array(self.xtx)
        xty = _owned_array(self.xty)
        if xtx.ndim != 3 or xtx.shape[1] != xtx.shape[2]:
            msg = "xtx must have shape (nodes, features, features)."
            raise ValueError(msg)
        if xty.shape != xtx.shape[:2]:
            msg = "xty must have shape (nodes, features), matching xtx."
            raise ValueError(msg)
        if not isfinite(self.yty) or self.yty < 0:
            msg = "yty must be finite and nonnegative."
            raise ValueError(msg)
        object.__setattr__(self, "xtx", xtx)
        object.__setattr__(self, "xty", xty)
        object.__setattr__(self, "yty", float(self.yty))


class NumericalProblem(Protocol):
    @property
    def task(self) -> Literal["regression", "binary"]: ...

    @property
    def axes(self) -> tuple[Axis, ...]: ...

    @property
    def axis_sizes(self) -> tuple[int, ...]: ...

    @property
    def features(self) -> tuple[str, ...]: ...

    @property
    def penalties(self) -> tuple[L1 | L2 | Laplacian, ...]: ...

    @property
    def n_samples(self) -> int: ...

    @property
    def n_nodes(self) -> int: ...

    @property
    def n_features(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    @property
    def design_fingerprint(self) -> str: ...

    @property
    def statistics(self) -> QuadraticStatistics | None: ...

    def iter_numeric(self) -> Iterator[NumericBatch]: ...


@dataclass(frozen=True, kw_only=True)
class ADMM:
    max_iterations: int = 1000
    atol: float = 1e-5
    rtol: float = 1e-4
    rho: float = 1.0
    adaptive_iterations: int = 50
    inner_max_iterations: int = 50
    inner_cg_iterations: int = 200
    max_passes: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_iterations", "inner_max_iterations", "inner_cg_iterations"):
            _count(getattr(self, name), name)
        for name in ("atol", "rtol", "rho"):
            _positive(getattr(self, name), name)
        if (
            isinstance(self.adaptive_iterations, bool)
            or not isinstance(self.adaptive_iterations, int)
            or self.adaptive_iterations < 0
        ):
            msg = "adaptive_iterations must be a nonnegative integer."
            raise ValueError(msg)
        if self.max_passes is not None:
            _count(self.max_passes, "max_passes")


@dataclass(frozen=True, kw_only=True)
class DirectSolve:
    max_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        _count(self.max_bytes, "max_bytes")


@dataclass(frozen=True, kw_only=True)
class ConjugateGradient:
    tol: float = 1e-9
    max_iterations: int = 1000
    max_passes: int | None = None

    def __post_init__(self) -> None:
        _positive(self.tol, "tol")
        _count(self.max_iterations, "max_iterations")
        if self.max_passes is not None:
            _count(self.max_passes, "max_passes")


@dataclass(frozen=True, kw_only=True)
class QuadraticSolver:
    linear_solver: DirectSolve | ConjugateGradient = field(default_factory=DirectSolve)


@dataclass(frozen=True, kw_only=True)
class Newton:
    max_iterations: int = 50
    atol: float = 1e-7
    rtol: float = 1e-7
    cg_max_iterations: int = 200
    max_passes: int | None = None

    def __post_init__(self) -> None:
        _count(self.max_iterations, "max_iterations")
        _count(self.cg_max_iterations, "cg_max_iterations")
        _positive(self.atol, "atol")
        _positive(self.rtol, "rtol")
        if self.max_passes is not None:
            _count(self.max_passes, "max_passes")


@dataclass(frozen=True, kw_only=True)
class AutoSolver:
    direct_max_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        _count(self.direct_max_bytes, "direct_max_bytes")


@dataclass(frozen=True, kw_only=True)
class CVXPYSolver:
    max_variables: int = 2000
    max_samples: int = 20000
    solver: str | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        _count(self.max_variables, "max_variables")
        _count(self.max_samples, "max_samples")


NewtonSolver = Newton

Solver = (
    ADMM
    | DirectSolve
    | ConjugateGradient
    | QuadraticSolver
    | Newton
    | AutoSolver
    | CVXPYSolver
)


@dataclass(frozen=True, kw_only=True)
class CacheStatistics:
    hits: int = 0
    misses: int = 0
    factorizations: int = 0
    bytes_used: int = 0
    entries: int = 0
    evictions: int = 0

    @property
    def factorization_count(self) -> int:
        return self.factorizations


@dataclass(frozen=True, kw_only=True)
class ObjectiveComponents:
    loss: float
    l1: float = 0.0
    l2: float = 0.0
    laplacian: float = 0.0
    penalties: tuple[tuple[str, float], ...] = ()

    @property
    def total(self) -> float:
        return self.loss + self.l1 + self.l2 + self.laplacian


@dataclass(frozen=True, kw_only=True)
class IterationRecord:
    iteration: int
    objective: float
    stationarity: float = float("inf")
    primal_residual: float | None = None
    dual_residual: float | None = None
    primal_threshold: float | None = None
    dual_threshold: float | None = None
    inner_residual: float | None = None
    inner_threshold: float | None = None
    rho: tuple[float, ...] = ()
    source_passes: int = 0
    rows_scanned: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, kw_only=True)
class Diagnostics:
    solver: str
    selection_reason: str = ""
    converged: bool = False
    termination: str = ""
    objective: ObjectiveComponents | None = None
    stationarity: float = float("inf")
    iterations: int = 0
    elapsed_seconds: float = 0.0
    source_passes: int = 0
    rows_scanned: int = 0
    cache: CacheStatistics = field(default_factory=CacheStatistics)
    history: tuple[IterationRecord, ...] = ()


@dataclass(frozen=True, kw_only=True)
class FailureReason:
    code: str
    message: str


@dataclass(frozen=True, kw_only=True)
class ADMMContinuation:
    fingerprint: str
    x: FloatArray
    u: FloatArray
    z: FloatArray
    rho: FloatArray

    def __post_init__(self) -> None:
        for name in ("x", "u", "z", "rho"):
            object.__setattr__(self, name, _owned_array(getattr(self, name)))
        if self.x.shape != self.u.shape or self.x.shape[1:] != self.z.shape:
            msg = "ADMM primal and dual state shapes do not match."
            raise ValueError(msg)
        if self.rho.shape != self.x.shape[:1] or np.any(self.rho <= 0):
            msg = "ADMM rho must be positive with one value per block."
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class NumericalSuccess:
    coefficients: FloatArray
    diagnostics: Diagnostics
    state: ADMMContinuation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", _owned_array(self.coefficients))


@dataclass(frozen=True, kw_only=True)
class NumericalFailure:
    reason: FailureReason
    diagnostics: Diagnostics


NumericalResult = NumericalSuccess | NumericalFailure
