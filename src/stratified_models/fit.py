"""Functional fitting and explicitly owned state for repeated solves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import cast, overload

import attrs
import numpy as np
from numpy.typing import NDArray

from stratified_models.data import TrainingSource, readonly_float_array
from stratified_models.errors import CoordinateError, ValidationError
from stratified_models.predict import (
    FittedBinaryClassifier,
    FittedPredictor,
    FittedRegressor,
)
from stratified_models.prepared import (
    PreparationDiagnostics,
    PreparationOptions,
    PreparedProblem,
    coordinate_key,
    prepare,
    prepare_batches,
)
from stratified_models.preprocessing import (
    FittedAddIntercept,
    FittedFixedWidthBins,
    FittedStandardize,
    FittedTransformChain,
)
from stratified_models.solvers import SolverCache, solve
from stratified_models.solvers.types import (
    ADMMContinuation,
    Diagnostics,
    FailureReason,
    NumericalFailure,
    Solver,
)
from stratified_models.specs import (
    BinaryLogisticSpec,
    ModelSpec,
    Pipeline,
    RegressionSpec,
    WeightRef,
)


@attrs.frozen(kw_only=True)
class FitSuccess[T]:
    predictor: T
    diagnostics: Diagnostics
    preparation: PreparationDiagnostics | None = None


@attrs.frozen(kw_only=True)
class FitFailure:
    reason: FailureReason
    diagnostics: Diagnostics
    preparation: PreparationDiagnostics | None = None


type FitResult[T] = FitSuccess[T] | FitFailure
type WeightOverrides = (
    Mapping[str, float] | Mapping[WeightRef, float] | Mapping[str | WeightRef, float]
)


@attrs.frozen(kw_only=True, eq=False)
class InitialGuess:
    """A solver-independent primal snapshot with explicit coordinate meaning."""

    coefficients: NDArray[np.float64] = attrs.field(converter=readonly_float_array)
    model: ModelSpec
    transformers: FittedTransformChain
    provenance: str = "explicit"

    def __attrs_post_init__(self) -> None:
        expected = (
            int(np.prod([axis.size for axis in self.model.axes])),
            len(self.model.features),
        )
        if (
            self.coefficients.shape != expected
            or not np.isfinite(self.coefficients).all()
        ):
            raise CoordinateError(
                f"initial coefficients must be finite with shape {expected}"
            )

    @property
    def coordinates(self) -> str:
        return coordinate_key(self.model, self.transformers)

    @classmethod
    def from_predictor(cls, predictor: FittedPredictor) -> InitialGuess:
        return cls(
            coefficients=predictor.coefficients,
            model=predictor.pipeline.model,
            transformers=predictor.transformers,
            provenance="fitted_predictor",
        )

    def transport_to(self, prepared: PreparedProblem[ModelSpec]) -> InitialGuess:
        target = prepared.pipeline.model
        if (
            self.model.task != target.task
            or self.model.features != target.features
            or self.model.axes != target.axes
        ):
            raise CoordinateError(
                "warm starts require matching task, feature identities and ordered strata"
            )
        if self.coordinates == prepared.coordinates:
            return self
        names = (*self.model.features, *(axis.name for axis in self.model.axes))
        old_origins = _coordinate_origins(self.transformers, names)
        new_origins = _coordinate_origins(prepared.transformers, names)
        if any(
            old_origins[axis.name] != new_origins[axis.name] for axis in self.model.axes
        ):
            raise CoordinateError("changed preprocessing changes stratum meaning")
        if any(
            _nonlinear_basis(old_origins[name]) != _nonlinear_basis(new_origins[name])
            for name in self.model.features
        ):
            raise CoordinateError("changed preprocessing changes feature meaning")
        old_a, old_b = _affine(self.transformers, self.model.features)
        new_a, new_b = _affine(prepared.transformers, target.features)
        variable = new_a != 0
        if np.any((~variable) & (old_a != 0) & np.any(self.coefficients != 0, axis=0)):
            raise CoordinateError(
                "the new feature basis cannot represent the initial predictor"
            )
        coefficients = np.zeros_like(self.coefficients)
        coefficients[:, variable] = (
            self.coefficients[:, variable] * old_a[variable] / new_a[variable]
        )
        offset = self.coefficients @ old_b - coefficients @ new_b
        constants = np.flatnonzero((~variable) & (new_b != 0))
        if len(constants):
            coefficients[:, constants[0]] = offset / new_b[constants[0]]
        elif not np.allclose(offset, 0, atol=1e-12, rtol=0):
            raise CoordinateError(
                "centering changes require a compatible intercept feature"
            )
        return InitialGuess(
            coefficients=coefficients,
            model=target,
            transformers=prepared.transformers,
            provenance=f"transported:{self.provenance}",
        )


type _Origin = tuple[object, ...]


def _coordinate_origins(
    transformers: FittedTransformChain, names: tuple[str, ...]
) -> dict[str, _Origin]:
    origins: dict[str, _Origin] = {}

    def origin(name: str) -> _Origin:
        return origins.get(name, ("raw", name))

    for step in transformers.steps:
        if isinstance(step, FittedStandardize):
            for name, mean, scale in zip(
                step.columns, step.means, step.scales, strict=True
            ):
                origins[name] = ("standardize", origin(name), float(mean), float(scale))
        elif isinstance(step, FittedAddIntercept):
            origins[step.name] = ("constant", 1.0)
        elif isinstance(step, FittedFixedWidthBins):
            origins[step.output] = (
                "bins",
                origin(step.column),
                step.origin,
                step.width,
                step.n_bins,
            )
        else:
            raise CoordinateError(
                "automatic transport requires known transform dependencies"
            )
    return {name: origin(name) for name in names}


def _nonlinear_basis(origin: _Origin) -> _Origin:
    # Only outer affine coordinate changes can be absorbed into coefficients.
    # Standardization inside a binning operation changes the bin's meaning.
    while origin[0] == "standardize":
        origin = cast(_Origin, origin[1])
    return origin


def _affine(
    transformers: FittedTransformChain, features: tuple[str, ...]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a, b = np.ones(len(features)), np.zeros(len(features))
    for step in transformers.steps:
        if isinstance(step, FittedStandardize):
            for name, mean, scale in zip(
                step.columns, step.means, step.scales, strict=True
            ):
                if name in features:
                    index = features.index(name)
                    a[index] /= scale
                    b[index] = (b[index] - mean) / scale
        elif isinstance(step, FittedAddIntercept):
            if step.name in features:
                index = features.index(step.name)
                a[index], b[index] = 0, 1
        elif not isinstance(step, FittedFixedWidthBins):
            raise CoordinateError(
                "automatic warm-start transport supports built-in affine transforms only"
            )
    return a, b


def _weights(
    model: ModelSpec, overrides: WeightOverrides | None
) -> Mapping[str, float]:
    result = {penalty.name: penalty.weight for penalty in model.penalties}
    seen: set[str] = set()
    if overrides is not None:
        for reference, value in overrides.items():
            name = reference.name if isinstance(reference, WeightRef) else reference
            if name in seen or name not in result:
                raise ValidationError(f"unknown or duplicate penalty override {name!r}")
            seen.add(name)
            weight = float(value)
            if not np.isfinite(weight) or weight < 0:
                raise ValidationError(
                    "penalty overrides must be finite and nonnegative"
                )
            result[name] = weight
    return MappingProxyType(result)


class SolveSession[M: ModelSpec]:
    """One prepared snapshot's warm starts, using an independently shared cache."""

    def __init__(
        self,
        prepared: PreparedProblem[M],
        *,
        solver: Solver,
        cache: SolverCache | None = None,
    ) -> None:
        self._prepared = prepared
        self._solver = solver
        self._cache = SolverCache() if cache is None else cache
        self._initial: NDArray[np.float64] | None = None
        self._state: ADMMContinuation | None = None

    @property
    def prepared(self) -> PreparedProblem[M]:
        return self._prepared

    @property
    def cache(self) -> SolverCache:
        return self._cache

    @overload
    def solve(
        self: SolveSession[RegressionSpec],
        *,
        weight_overrides: WeightOverrides | None = None,
        initial: InitialGuess | None = None,
    ) -> FitResult[FittedRegressor]: ...

    @overload
    def solve(
        self: SolveSession[BinaryLogisticSpec],
        *,
        weight_overrides: WeightOverrides | None = None,
        initial: InitialGuess | None = None,
    ) -> FitResult[FittedBinaryClassifier]: ...

    @overload
    def solve(
        self,
        *,
        weight_overrides: WeightOverrides | None = None,
        initial: InitialGuess | None = None,
    ) -> FitResult[FittedPredictor]: ...

    def solve(
        self,
        *,
        weight_overrides: WeightOverrides | None = None,
        initial: InitialGuess | None = None,
    ) -> FitResult[FittedPredictor]:
        weights = _weights(self.prepared.pipeline.model, weight_overrides)
        if initial is not None and initial.coordinates != self.prepared.coordinates:
            raise CoordinateError(
                "initial guess coordinates differ; explicitly transport_to(prepared) first"
            )
        coefficients = self._initial if initial is None else initial.coefficients
        state = self._state if initial is None else None
        outcome = solve(
            self.prepared,
            solver=self._solver,
            weights=weights,
            cache=self.cache,
            initial=None if state is not None else coefficients,
            state=state,
        )
        if isinstance(outcome, NumericalFailure):
            return FitFailure(
                reason=outcome.reason,
                diagnostics=outcome.diagnostics,
                preparation=self.prepared.preparation,
            )
        problem = self.prepared
        if isinstance(problem.pipeline.model, RegressionSpec):
            predictor: FittedPredictor = FittedRegressor(
                pipeline=cast(Pipeline[RegressionSpec], problem.pipeline),
                transformers=problem.transformers,
                coefficients=outcome.coefficients,
                weights=weights,
                training_diagnostics=outcome.diagnostics,
            )
        else:
            predictor = FittedBinaryClassifier(
                pipeline=cast(Pipeline[BinaryLogisticSpec], problem.pipeline),
                transformers=problem.transformers,
                coefficients=outcome.coefficients,
                weights=weights,
                training_diagnostics=outcome.diagnostics,
            )
        self._initial = outcome.coefficients
        self._state = outcome.state
        return FitSuccess(
            predictor=predictor,
            diagnostics=outcome.diagnostics,
            preparation=self.prepared.preparation,
        )


@overload
def fit(
    pipeline: Pipeline[RegressionSpec],
    X: object,
    y: object,
    *,
    solver: Solver,
    options: PreparationOptions | None = None,
    cache: SolverCache | None = None,
    feature_names: Sequence[str] | None = None,
) -> FitResult[FittedRegressor]: ...


@overload
def fit(
    pipeline: Pipeline[BinaryLogisticSpec],
    X: object,
    y: object,
    *,
    solver: Solver,
    options: PreparationOptions | None = None,
    cache: SolverCache | None = None,
    feature_names: Sequence[str] | None = None,
) -> FitResult[FittedBinaryClassifier]: ...


def fit(
    pipeline: Pipeline[ModelSpec],
    X: object,
    y: object,
    *,
    solver: Solver,
    options: PreparationOptions | None = None,
    cache: SolverCache | None = None,
    feature_names: Sequence[str] | None = None,
) -> FitResult[FittedPredictor]:
    prepared = prepare(pipeline, X, y, options=options, feature_names=feature_names)
    return SolveSession(prepared, solver=solver, cache=cache).solve()


@overload
def fit_batches(
    pipeline: Pipeline[RegressionSpec],
    source: TrainingSource,
    *,
    solver: Solver,
    options: PreparationOptions | None = None,
    cache: SolverCache | None = None,
) -> FitResult[FittedRegressor]: ...


@overload
def fit_batches(
    pipeline: Pipeline[BinaryLogisticSpec],
    source: TrainingSource,
    *,
    solver: Solver,
    options: PreparationOptions | None = None,
    cache: SolverCache | None = None,
) -> FitResult[FittedBinaryClassifier]: ...


def fit_batches(
    pipeline: Pipeline[ModelSpec],
    source: TrainingSource,
    *,
    solver: Solver,
    options: PreparationOptions | None = None,
    cache: SolverCache | None = None,
) -> FitResult[FittedPredictor]:
    return SolveSession(
        prepare_batches(pipeline, source, options=options), solver=solver, cache=cache
    ).solve()


fit_arrays = fit
