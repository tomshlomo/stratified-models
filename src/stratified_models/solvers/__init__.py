"""Explicit numerical solver strategies and bounded reuse."""

from .cache import SolverCache
from .engine import solve
from .types import (
    ADMM,
    ADMMContinuation,
    AutoSolver,
    CacheStatistics,
    ConjugateGradient,
    CVXPYSolver,
    Diagnostics,
    DirectSolve,
    FailureReason,
    IterationRecord,
    Newton,
    NewtonSolver,
    NumericalFailure,
    NumericalProblem,
    NumericalSuccess,
    ObjectiveComponents,
    QuadraticSolver,
    QuadraticStatistics,
    Solver,
)

__all__ = [
    "ADMM",
    "ADMMContinuation",
    "AutoSolver",
    "CVXPYSolver",
    "CacheStatistics",
    "ConjugateGradient",
    "Diagnostics",
    "DirectSolve",
    "FailureReason",
    "IterationRecord",
    "Newton",
    "NewtonSolver",
    "NumericalFailure",
    "NumericalProblem",
    "NumericalSuccess",
    "ObjectiveComponents",
    "QuadraticSolver",
    "QuadraticStatistics",
    "Solver",
    "SolverCache",
    "solve",
]
