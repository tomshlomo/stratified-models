"""CPU orchestration with bounded, reusable JAX sample kernels."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Hashable, Iterator, Mapping
from dataclasses import replace
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, cg

from stratified_models.data import NumericBatch
from stratified_models.errors import OptionalDependencyError, ValidationError
from stratified_models.specs import L1, L2, Laplacian

from .cache import SolverCache
from .types import (
    ADMM,
    ADMMContinuation,
    AutoSolver,
    ConjugateGradient,
    CVXPYSolver,
    Diagnostics,
    DirectSolve,
    FailureReason,
    IterationRecord,
    Newton,
    NumericalFailure,
    NumericalProblem,
    NumericalSuccess,
    ObjectiveComponents,
    QuadraticSolver,
    QuadraticStatistics,
    Solver,
)

Array = NDArray[np.float64]
Factor = (
    tuple[Literal["cholesky"], tuple[Array, bool]]
    | tuple[Literal["eigen"], tuple[Array, Array]]
)
_BATCH_SIZE = 256


@jax.jit
def _regression_batch(
    theta: jax.Array, x: jax.Array, y: jax.Array, nodes: jax.Array, mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    prediction = jnp.sum(theta[nodes] * x, axis=1)
    residual = (prediction - y) * mask
    value = jnp.sum(jnp.square(residual)) / 2
    gradient = jnp.zeros_like(theta).at[nodes].add(residual[:, None] * x)
    return value, gradient


@jax.jit
def _logistic_batch(
    theta: jax.Array, x: jax.Array, y: jax.Array, nodes: jax.Array, mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    prediction = jnp.sum(theta[nodes] * x, axis=1)
    signed = (2 * y - 1) * prediction
    value = jnp.sum(jnp.logaddexp(0.0, -signed) * mask)
    residual = (jax.nn.sigmoid(prediction) - y) * mask
    gradient = jnp.zeros_like(theta).at[nodes].add(residual[:, None] * x)
    return value, gradient


@jax.jit
def _regression_hvp(
    theta: jax.Array,
    vector: jax.Array,
    x: jax.Array,
    y: jax.Array,
    nodes: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    del theta, y
    product = jnp.sum(vector[nodes] * x, axis=1) * mask
    return jnp.zeros_like(vector).at[nodes].add(product[:, None] * x)


@jax.jit
def _logistic_hvp(
    theta: jax.Array,
    vector: jax.Array,
    x: jax.Array,
    y: jax.Array,
    nodes: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    del y
    probability = jax.nn.sigmoid(jnp.sum(theta[nodes] * x, axis=1))
    product = jnp.sum(vector[nodes] * x, axis=1)
    product = product * probability * (1 - probability) * mask
    return jnp.zeros_like(vector).at[nodes].add(product[:, None] * x)


class _NumericalError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _MatrixFreeOperator(LinearOperator):
    def __init__(self, size: int, apply: Callable[[Array], Array]) -> None:
        self._apply = apply
        super().__init__(dtype=np.dtype(np.float64), shape=(size, size))

    def _matvec(self, x: Array) -> Array:
        return self._apply(x)


class _Evaluation:
    def __init__(self, problem: NumericalProblem, max_passes: int | None) -> None:
        self.problem = problem
        self.max_passes = max_passes
        self.passes = 0
        self.rows = 0
        self.statistics = problem.statistics

    def batches(self) -> Iterator[NumericBatch]:
        if self.max_passes is not None and self.passes >= self.max_passes:
            raise _NumericalError(
                "pass_limit", "The configured source-pass budget was exhausted"
            )
        self.passes += 1
        iterator = self.problem.iter_numeric()
        try:
            for batch in iterator:
                self.rows += len(batch.y)
                yield batch
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def _chunks(self) -> Iterator[tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        for batch in self.batches():
            for first in range(0, len(batch.y), _BATCH_SIZE):
                count = min(_BATCH_SIZE, len(batch.y) - first)
                x = np.zeros((_BATCH_SIZE, self.problem.n_features), dtype=np.float64)
                y = np.zeros(_BATCH_SIZE, dtype=np.float64)
                nodes = np.zeros(_BATCH_SIZE, dtype=np.int64)
                mask = np.zeros(_BATCH_SIZE, dtype=np.float64)
                x[:count] = batch.x[first : first + count]
                y[:count] = batch.y[first : first + count]
                nodes[:count] = batch.nodes[first : first + count]
                mask[:count] = 1
                yield (
                    jnp.asarray(x),
                    jnp.asarray(y),
                    jnp.asarray(nodes),
                    jnp.asarray(mask),
                )

    def value_gradient(self, theta: Array) -> tuple[float, Array]:
        if self.problem.task == "regression" and self.statistics is not None:
            stats = self.statistics
            product = np.einsum("kij,kj->ki", stats.xtx, theta)
            value = (
                np.sum(theta * product) / 2 - np.sum(stats.xty * theta) + stats.yty / 2
            )
            return float(value / self.problem.n_samples), (
                product - stats.xty
            ) / self.problem.n_samples
        function = (
            _regression_batch if self.problem.task == "regression" else _logistic_batch
        )
        value = 0.0
        gradient = np.zeros_like(theta)
        for x, y, nodes, mask in self._chunks():
            batch_value, batch_gradient = function(
                jnp.asarray(theta), x, y, nodes, mask
            )
            # Synchronizing each bounded chunk also bounds pending device work.
            value += float(batch_value)
            gradient += np.asarray(batch_gradient)
        return value / self.problem.n_samples, gradient / self.problem.n_samples

    def hvp(self, theta: Array, vector: Array) -> Array:
        if self.problem.task == "regression" and self.statistics is not None:
            return (
                np.einsum("kij,kj->ki", self.statistics.xtx, vector)
                / self.problem.n_samples
            )
        function = (
            _regression_hvp if self.problem.task == "regression" else _logistic_hvp
        )
        product = np.zeros_like(vector)
        for x, y, nodes, mask in self._chunks():
            product += np.asarray(
                function(jnp.asarray(theta), jnp.asarray(vector), x, y, nodes, mask)
            )
        return product / self.problem.n_samples

    def collect_statistics(self) -> QuadraticStatistics:
        if self.statistics is not None:
            return self.statistics
        p = self.problem
        xtx = np.zeros((p.n_nodes, p.n_features, p.n_features))
        xty = np.zeros((p.n_nodes, p.n_features))
        yty = 0.0
        for batch in self.batches():
            # Avoid a sample-by-feature-by-feature temporary even for large batches.
            for first in range(0, len(batch.y), _BATCH_SIZE):
                nodes = batch.nodes[first : first + _BATCH_SIZE]
                for node in np.unique(nodes):
                    selected = nodes == node
                    x = batch.x[first : first + _BATCH_SIZE][selected]
                    y = batch.y[first : first + _BATCH_SIZE][selected]
                    xtx[node] += x.T @ x
                    xty[node] += x.T @ y
            yty += float(batch.y @ batch.y)
        self.statistics = QuadraticStatistics(xtx=xtx, xty=xty, yty=yty)
        return self.statistics


class _Penalties:
    def __init__(self, problem: NumericalProblem, weights: Mapping[str, float]) -> None:
        self.problem = problem
        known = {penalty.name for penalty in problem.penalties}
        unknown = set(weights) - known
        if unknown:
            raise ValidationError(f"Unknown penalty weights: {sorted(unknown)}")
        self.weights = {
            penalty.name: float(weights.get(penalty.name, penalty.weight))
            for penalty in problem.penalties
        }
        if any(
            not math.isfinite(value) or value < 0 for value in self.weights.values()
        ):
            raise ValidationError("Penalty weights must be finite and nonnegative")
        self.l1 = np.zeros(problem.n_features)
        self.l2 = np.zeros(problem.n_features)
        self.graphs: list[tuple[Laplacian, int, Array, float]] = []
        names = tuple(problem.features)
        axes = tuple(axis.name for axis in problem.axes)
        for penalty in problem.penalties:
            mask = np.array(
                [name in penalty.features for name in names], dtype=np.float64
            )
            weight = self.weights[penalty.name]
            if isinstance(penalty, L1):
                self.l1 += weight * mask
            elif isinstance(penalty, L2):
                self.l2 += weight * mask
            elif isinstance(penalty, Laplacian):
                self.graphs.append(
                    (penalty, axes.index(penalty.graph.axis.name), mask, weight)
                )
        self.graph_key = tuple(
            (
                p.name,
                p.graph.axis.name,
                p.graph.axis.nodes,
                p.graph.edges,
                p.features,
                w,
            )
            for p, _, _, w in self.graphs
        )
        self.has_active_graph = any(
            weight > 0
            and any(
                edge_weight > 0 for _, _, edge_weight in penalty.graph.indexed_edges
            )
            for penalty, _, _, weight in self.graphs
        )

    def graph_product(self, theta: Array) -> Array:
        shape = (*self.problem.axis_sizes, self.problem.n_features)
        original = theta.reshape(shape)
        output = np.zeros(shape, dtype=np.float64)
        for penalty, axis, mask, weight in self.graphs:
            if weight == 0:
                continue
            for left, right, edge_weight in penalty.graph.indexed_edges:
                a = [slice(None)] * len(shape)
                b = list(a)
                a[axis], b[axis] = left, right
                aa, bb = tuple(a), tuple(b)
                difference = (original[aa] - original[bb]) * mask
                output[aa] += weight * edge_weight * difference
                output[bb] -= weight * edge_weight * difference
        return output.reshape(theta.shape)

    def components(self, theta: Array, loss: float) -> ObjectiveComponents:
        terms = []
        l1 = l2 = laplacian = 0.0
        for penalty in self.problem.penalties:
            columns = [self.problem.features.index(name) for name in penalty.features]
            if isinstance(penalty, L1):
                value = self.weights[penalty.name] * float(
                    np.abs(theta[:, columns]).sum()
                )
                l1 += value
            elif isinstance(penalty, L2):
                value = (
                    self.weights[penalty.name]
                    * float(np.square(theta[:, columns]).sum())
                    / 2
                )
                l2 += value
            else:
                # This uses each named graph independently, including feature masks.
                graph_value = 0.0
                original = theta.reshape(
                    (*self.problem.axis_sizes, self.problem.n_features)
                )
                axis = tuple(a.name for a in self.problem.axes).index(
                    penalty.graph.axis.name
                )
                for left, right, edge_weight in penalty.graph.indexed_edges:
                    difference = np.take(original, left, axis=axis) - np.take(
                        original, right, axis=axis
                    )
                    graph_value += (
                        edge_weight
                        * float(np.square(difference[..., columns]).sum())
                        / 2
                    )
                value = self.weights[penalty.name] * graph_value
                laplacian += value
            terms.append((penalty.name, value))
        return ObjectiveComponents(
            loss=loss, l1=l1, l2=l2, laplacian=laplacian, penalties=tuple(terms)
        )

    def smooth_gradient(self, theta: Array, loss_gradient: Array) -> Array:
        return loss_gradient + self.l2 * theta + self.graph_product(theta)

    def stationarity(self, theta: Array, loss_gradient: Array) -> float:
        gradient = self.smooth_gradient(theta, loss_gradient)
        return float(np.linalg.norm(theta - _soft(theta - gradient, self.l1)))


def _soft(value: Array, threshold: Array | float) -> Array:
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0)


def _finite(value: Array, description: str) -> None:
    if not np.all(np.isfinite(value)):
        raise _NumericalError("nonfinite", f"Nonfinite {description}")


def _cg_solve(
    operator: Callable[[Array], Array],
    b: Array,
    initial: Array,
    tolerance: float,
    max_iterations: int,
) -> tuple[Array, float, int]:
    shape = b.shape
    iterations = 0

    def matvec(vector: Array) -> Array:
        return operator(vector.reshape(shape)).ravel()

    def callback(_: Array) -> None:
        nonlocal iterations
        iterations += 1

    linear = _MatrixFreeOperator(b.size, matvec)
    value, info = cg(
        linear,
        b.ravel(),
        x0=initial.ravel(),
        rtol=0,
        atol=max(tolerance, 1e-14),
        maxiter=max_iterations,
        callback=callback,
    )
    value = value.reshape(shape)
    _finite(value, "CG solution")
    residual = float(np.linalg.norm(operator(value) - b))
    if info < 0:
        raise _NumericalError("linear_breakdown", "Conjugate gradient broke down")
    return value, residual, iterations


def _smooth_newton(
    value_gradient: Callable[[Array], tuple[float, Array]],
    product: Callable[[Array, Array], Array],
    initial: Array,
    *,
    tolerance: float,
    max_iterations: int,
    cg_iterations: int,
    record: Callable[[int, Array, float, Array], None] | None = None,
) -> tuple[Array, float, int]:
    theta = initial.copy()
    for iteration in range(max_iterations + 1):
        value, gradient = value_gradient(theta)
        _finite(gradient, "gradient")
        if not math.isfinite(value):
            raise _NumericalError("nonfinite", "Nonfinite objective")
        residual = float(np.linalg.norm(gradient))
        if record is not None:
            record(iteration, theta, value, gradient)
        if residual <= tolerance:
            return theta, residual, iteration
        if iteration == max_iterations:
            break
        # The tiny numerical damping affects the search direction, not the objective.
        step, _, _ = _cg_solve(
            lambda v, point=theta: product(point, v) + 1e-12 * v,
            gradient,
            np.zeros_like(theta),
            max(tolerance * 0.1, min(0.1, math.sqrt(residual)) * residual),
            cg_iterations,
        )
        decrement = float(np.sum(gradient * step))
        if not math.isfinite(decrement) or decrement <= 0:
            step, decrement = gradient, float(np.sum(gradient * gradient))
        alpha = 1.0
        for _ in range(40):
            candidate = theta - alpha * step
            candidate_value, _ = value_gradient(candidate)
            if math.isfinite(
                candidate_value
            ) and candidate_value <= value - 1e-4 * alpha * decrement + 1e-15 * max(
                1.0, abs(value)
            ):
                theta = candidate
                break
            alpha *= 0.5
        else:
            raise _NumericalError(
                "line_search_failed",
                "Newton line search failed to find a finite descent step",
            )
    return theta, residual, max_iterations


class _Run:
    def __init__(
        self,
        problem: NumericalProblem,
        penalties: _Penalties,
        cache: SolverCache,
        method: str,
        reason: str,
        max_passes: int | None,
    ) -> None:
        self.problem = problem
        self.penalties = penalties
        self.cache = cache
        self.method = method
        self.reason = reason
        self.evaluation = _Evaluation(problem, max_passes)
        self.started = time.perf_counter()
        self.before_cache = cache.snapshot()
        self.history: list[IterationRecord] = []
        self.objective: ObjectiveComponents | None = None
        self.stationarity = math.inf
        self.iterations = 0

    def assess(self, theta: Array) -> Array:
        loss, gradient = self.evaluation.value_gradient(theta)
        self.objective = self.penalties.components(theta, loss)
        self.stationarity = self.penalties.stationarity(theta, gradient)
        return gradient

    def diagnostics(self, converged: bool, termination: str) -> Diagnostics:
        after = self.cache.snapshot()
        before = self.before_cache
        counts = replace(
            after,
            hits=after.hits - before.hits,
            misses=after.misses - before.misses,
            factorizations=after.factorizations - before.factorizations,
        )
        return Diagnostics(
            solver=self.method,
            selection_reason=self.reason,
            converged=converged,
            termination=termination,
            objective=self.objective,
            stationarity=self.stationarity,
            iterations=self.iterations,
            elapsed_seconds=time.perf_counter() - self.started,
            source_passes=self.evaluation.passes,
            rows_scanned=self.evaluation.rows,
            cache=counts,
            history=tuple(self.history),
        )


def _factor(
    cache: SolverCache, key: Hashable, build_matrix: Callable[[], Array]
) -> Factor:
    cached = cache.get(key)
    if cached is not None:
        return cast("Factor", cached)
    matrix = build_matrix()
    try:
        factor: Factor = ("cholesky", linalg.cho_factor(matrix, check_finite=True))
    except linalg.LinAlgError:
        eigenvalues, vectors = linalg.eigh(matrix)
        factor = ("eigen", (eigenvalues, vectors))
    cache.record_factorization()
    cache.put(key, factor, nbytes=matrix.nbytes + matrix.shape[0] * 8)
    return factor


def _factor_solve(factor: Factor, right: Array) -> Array:
    kind, data = factor
    if kind == "cholesky":
        return linalg.cho_solve(data, right)
    eigenvalues, vectors = cast("tuple[Array, Array]", data)
    cutoff = (
        np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(eigenvalues))))
        * len(eigenvalues)
    )
    if np.min(eigenvalues) < -100 * cutoff:
        raise _NumericalError(
            "indefinite", "The quadratic operator is not positive semidefinite"
        )
    inverse = np.zeros_like(eigenvalues)
    np.divide(1.0, eigenvalues, out=inverse, where=eigenvalues > cutoff)
    return vectors @ (inverse * (vectors.T @ right))


def _direct_bytes(problem: NumericalProblem, penalties: _Penalties) -> int:
    block_bytes = problem.n_nodes * problem.n_features**2 * 8
    if not penalties.has_active_graph:
        return 3 * block_bytes
    return 3 * (problem.n_nodes * problem.n_features) ** 2 * 8 + block_bytes


def _uncoupled_quadratic(run: _Run, statistics: QuadraticStatistics) -> Array:
    p, penalties = run.problem, run.penalties
    key = ("quadratic_blocks", p.design_fingerprint, p.n_samples, tuple(penalties.l2))
    cached = run.cache.get(key)
    if cached is None:
        matrices = statistics.xtx / p.n_samples + np.diag(penalties.l2)
        eigenvalues, vectors = np.linalg.eigh(matrices)
        # One batched factorization call, with one small matrix per stratum.
        run.cache.record_factorization()
        run.cache.put(key, (eigenvalues, vectors))
    else:
        eigenvalues, vectors = cast("tuple[Array, Array]", cached)
    cutoff = (
        np.finfo(np.float64).eps
        * np.maximum(1.0, np.max(np.abs(eigenvalues), axis=1))
        * p.n_features
    )
    if np.any(eigenvalues < -100 * cutoff[:, None]):
        raise _NumericalError(
            "indefinite", "A quadratic stratum operator is not positive semidefinite"
        )
    inverse = np.zeros_like(eigenvalues)
    np.divide(1.0, eigenvalues, out=inverse, where=eigenvalues > cutoff[:, None])
    right = statistics.xty / p.n_samples
    transformed = np.einsum("kji,kj->ki", vectors, right)
    # Null directions are assigned zero, selecting a canonical minimum-norm fit.
    return np.einsum("kij,kj->ki", vectors, inverse * transformed)


def _quadratic(
    run: _Run, config: DirectSolve | ConjugateGradient, theta: Array
) -> tuple[Array, None]:
    p, penalties, evaluation = run.problem, run.penalties, run.evaluation
    if p.task != "regression" or np.any(penalties.l1):
        raise ValidationError(
            "A quadratic solver requires squared loss and zero L1 weights"
        )
    if isinstance(config, DirectSolve):
        required = _direct_bytes(p, penalties)
        if required > config.max_bytes:
            raise _NumericalError(
                "memory_limit",
                f"Direct quadratic solve requires approximately {required} bytes, exceeding max_bytes={config.max_bytes}",
            )
        statistics = evaluation.collect_statistics()
        if not penalties.has_active_graph:
            theta = _uncoupled_quadratic(run, statistics)
        else:

            def build_matrix() -> Array:
                matrix = linalg.block_diag(
                    *(block / p.n_samples for block in statistics.xtx)
                )
                for column in range(theta.size):
                    basis = np.zeros_like(theta)
                    basis.ravel()[column] = 1
                    matrix[:, column] += penalties.graph_product(basis).ravel()
                matrix.flat[:: theta.size + 1] += np.broadcast_to(
                    penalties.l2, theta.shape
                ).ravel()
                return matrix

            key = (
                "quadratic",
                p.design_fingerprint,
                p.n_samples,
                p.axis_sizes,
                tuple(p.features),
                tuple(penalties.l2),
                penalties.graph_key,
            )
            factor = _factor(run.cache, key, build_matrix)
            theta = _factor_solve(
                factor, (statistics.xty / p.n_samples).ravel()
            ).reshape(theta.shape)
        iterations, tolerance = (
            1,
            1e-7 * (1 + np.linalg.norm(statistics.xty / p.n_samples)),
        )
    else:
        _, gradient_zero = evaluation.value_gradient(np.zeros_like(theta))
        right = -gradient_zero

        def operator(v: Array) -> Array:
            return (
                evaluation.hvp(np.zeros_like(theta), v)
                + penalties.l2 * v
                + penalties.graph_product(v)
            )

        tolerance = config.tol * max(1.0, float(np.linalg.norm(right)))
        theta, _, iterations = _cg_solve(
            operator, right, theta, tolerance, config.max_iterations
        )
    _finite(theta, "quadratic coefficients")
    run.iterations = iterations
    run.assess(theta)
    assert run.objective is not None
    run.history.append(
        IterationRecord(
            iteration=iterations,
            objective=run.objective.total,
            stationarity=run.stationarity,
            source_passes=evaluation.passes,
            rows_scanned=evaluation.rows,
            elapsed_seconds=time.perf_counter() - run.started,
        )
    )
    if run.stationarity > max(1e-7, 10 * tolerance):
        raise _NumericalError(
            "linear_limit", "The quadratic solver did not meet its optimality tolerance"
        )
    return theta, None


def _newton(run: _Run, config: Newton, theta: Array) -> tuple[Array, None]:
    penalties, evaluation = run.penalties, run.evaluation
    if np.any(penalties.l1):
        raise ValidationError(
            "Newton requires zero L1 weights; use ADMM for nonsmooth penalties"
        )

    def value_gradient(value: Array) -> tuple[float, Array]:
        loss, gradient = evaluation.value_gradient(value)
        return penalties.components(value, loss).total, penalties.smooth_gradient(
            value, gradient
        )

    def product(value: Array, vector: Array) -> Array:
        return (
            evaluation.hvp(value, vector)
            + penalties.l2 * vector
            + penalties.graph_product(vector)
        )

    _, initial_gradient = value_gradient(theta)
    tolerance = config.atol + config.rtol * float(np.linalg.norm(initial_gradient))

    def record(iteration: int, value: Array, objective: float, gradient: Array) -> None:
        run.iterations = iteration
        run.stationarity = float(np.linalg.norm(gradient))
        run.history.append(
            IterationRecord(
                iteration=iteration,
                objective=objective,
                stationarity=run.stationarity,
                source_passes=evaluation.passes,
                rows_scanned=evaluation.rows,
                elapsed_seconds=time.perf_counter() - run.started,
            )
        )

    theta, residual, _ = _smooth_newton(
        value_gradient,
        product,
        theta,
        tolerance=tolerance,
        max_iterations=config.max_iterations,
        cg_iterations=config.cg_max_iterations,
        record=record,
    )
    run.assess(theta)
    if residual > tolerance:
        raise _NumericalError(
            "iteration_limit",
            "Newton exhausted its iteration budget before convergence",
        )
    return theta, None


def _loss_prox(
    run: _Run, v: Array, rho: float, initial: Array, target: float, config: ADMM
) -> tuple[Array, float]:
    evaluation, p = run.evaluation, run.problem
    if p.task == "regression":
        if evaluation.statistics is not None:
            stats = evaluation.statistics
            key = ("loss_prox", p.design_fingerprint, p.n_samples, rho)
            factors = cast("Array | None", run.cache.get(key))
            if factors is None:
                matrices = stats.xtx / p.n_samples + rho * np.eye(p.n_features)
                factors = np.linalg.cholesky(matrices)
                run.cache.record_factorization()
                run.cache.put(key, factors)
            right = stats.xty / p.n_samples + rho * v
            theta = np.linalg.solve(
                np.swapaxes(factors, -1, -2), np.linalg.solve(factors, right[..., None])
            )[..., 0]
            residual = float(
                np.linalg.norm(
                    evaluation.hvp(theta, theta)
                    - stats.xty / p.n_samples
                    + rho * (theta - v)
                )
            )
            return theta, residual
        _, zero_gradient = evaluation.value_gradient(np.zeros_like(v))
        return _cg_solve(
            lambda value: evaluation.hvp(value, value) + rho * value,
            rho * v - zero_gradient,
            initial,
            target,
            config.inner_cg_iterations,
        )[:2]

    def value_gradient(value: Array) -> tuple[float, Array]:
        loss, gradient = evaluation.value_gradient(value)
        return loss + rho * float(np.square(value - v).sum()) / 2, gradient + rho * (
            value - v
        )

    theta, residual, _ = _smooth_newton(
        value_gradient,
        lambda value, vector: evaluation.hvp(value, vector) + rho * vector,
        initial,
        tolerance=target,
        max_iterations=config.inner_max_iterations,
        cg_iterations=config.inner_cg_iterations,
    )
    return theta, residual


def _admm(
    run: _Run, config: ADMM, theta: Array, continuation: ADMMContinuation | None
) -> tuple[Array, ADMMContinuation]:
    penalties, evaluation = run.penalties, run.evaluation
    if continuation is not None:
        if continuation.fingerprint != run.problem.fingerprint:
            raise ValidationError(
                "ADMM continuation belongs to a different data snapshot; transfer an explicit primal start instead"
            )
        x, u, z, rho = (
            np.array(value, copy=True)
            for value in (
                continuation.x,
                continuation.u,
                continuation.z,
                continuation.rho,
            )
        )
    else:
        z = theta.copy()
        x = np.stack((z, z, z))
        u = np.zeros_like(x)
        rho = np.full(3, config.rho)
    if (
        x.shape != (3, *theta.shape)
        or u.shape != x.shape
        or z.shape != theta.shape
        or rho.shape != (3,)
        or np.any(rho <= 0)
    ):
        raise ValidationError(
            "ADMM continuation state is incompatible with this problem"
        )
    for value in (x, u, z, rho):
        _finite(value, "ADMM initialization")
    previous_primal = previous_dual = None
    for iteration in range(1, config.max_iterations + 1):
        run.iterations = iteration
        if (
            iteration <= config.adaptive_iterations
            and iteration % 10 == 0
            and previous_primal is not None
        ):
            old_rho = rho.copy()
            increase = previous_primal > 10 * previous_dual
            decrease = previous_dual > 10 * previous_primal
            rho = np.clip(
                rho * np.where(increase, 2.0, np.where(decrease, 0.5, 1.0)), 1e-6, 1e6
            )
            u *= (old_rho / rho)[:, None, None]
        # The distance-to-exact-prox certificate is summably tightened.
        eta = 0.01 / iteration**1.5
        x[0], inner_loss = _loss_prox(run, z - u[0], rho[0], x[0], rho[0] * eta, config)
        if inner_loss > max(rho[0] * eta * 1.05, 1e-12):
            raise _NumericalError(
                "inner_limit",
                "The loss proximal solve did not meet its certified accuracy",
            )
        x[1] = _soft(z - u[1], penalties.l1 / rho[1]) / (1 + penalties.l2 / rho[1])
        if any(weight > 0 for _, _, _, weight in penalties.graphs):
            x[2], inner_graph, _ = _cg_solve(
                lambda value, graph_rho=float(rho[2]): graph_rho * value
                + penalties.graph_product(value),
                rho[2] * (z - u[2]),
                x[2],
                rho[2] * eta,
                config.inner_cg_iterations,
            )
            if inner_graph > max(rho[2] * eta * 1.05, 1e-12):
                raise _NumericalError(
                    "inner_limit",
                    "The graph proximal solve did not meet its certified accuracy",
                )
        else:
            x[2], inner_graph = z - u[2], 0.0
        old_z = z.copy()
        z = np.sum(rho[:, None, None] * (x + u), axis=0) / rho.sum()
        primal = x - z
        u += primal
        _finite(z, "ADMM coefficients")
        previous_primal = np.linalg.norm(primal.reshape(3, -1), axis=1)
        previous_dual = rho * np.linalg.norm(z - old_z)
        primal_norm = float(np.linalg.norm(previous_primal))
        dual_norm = float(np.linalg.norm(previous_dual))
        primal_threshold = math.sqrt(x.size) * config.atol + config.rtol * max(
            float(np.linalg.norm(x)), math.sqrt(3) * float(np.linalg.norm(z))
        )
        dual_threshold = math.sqrt(x.size) * config.atol + config.rtol * float(
            np.linalg.norm(rho[:, None, None] * u)
        )
        run.assess(z)
        assert run.objective is not None
        run.history.append(
            IterationRecord(
                iteration=iteration,
                objective=run.objective.total,
                stationarity=run.stationarity,
                primal_residual=primal_norm,
                dual_residual=dual_norm,
                primal_threshold=primal_threshold,
                dual_threshold=dual_threshold,
                inner_residual=max(inner_loss / rho[0], inner_graph / rho[2]),
                inner_threshold=eta,
                rho=tuple(float(value) for value in rho),
                source_passes=evaluation.passes,
                rows_scanned=evaluation.rows,
                elapsed_seconds=time.perf_counter() - run.started,
            )
        )
        stationarity_threshold = math.sqrt(z.size) * config.atol + config.rtol * max(
            1.0, float(np.linalg.norm(z))
        )
        if (
            primal_norm <= primal_threshold
            and dual_norm <= dual_threshold
            and run.stationarity <= stationarity_threshold
        ):
            return z, ADMMContinuation(
                fingerprint=run.problem.fingerprint, x=x, u=u, z=z, rho=rho
            )
    raise _NumericalError(
        "iteration_limit",
        "ADMM exhausted its iteration budget before satisfying residual and stationarity tolerances",
    )


def _cvxpy(run: _Run, config: CVXPYSolver, theta: Array) -> tuple[Array, None]:
    try:
        import cvxpy as cp
    except ImportError as error:
        raise OptionalDependencyError(
            "CVXPY references require the cvxpy integration extra"
        ) from error
    p, penalties = run.problem, run.penalties
    if theta.size > config.max_variables or p.n_samples > config.max_samples:
        raise _NumericalError(
            "memory_limit",
            "CVXPY reference exceeds its configured small-problem limits",
        )
    variable = cp.Variable(theta.shape)
    loss = cp.Constant(0.0)
    for batch in run.evaluation.batches():
        prediction = cp.sum(cp.multiply(variable[batch.nodes], batch.x), axis=1)
        if p.task == "regression":
            loss += cp.sum_squares(prediction - batch.y) / (2 * p.n_samples)
        else:
            loss += (
                cp.sum(cp.logistic(cp.multiply(1 - 2 * batch.y, prediction)))
                / p.n_samples
            )
    cost = (
        loss
        + cp.sum(cp.multiply(penalties.l1, cp.abs(variable)))
        + cp.sum(cp.multiply(penalties.l2, cp.square(variable))) / 2
    )
    for penalty, axis, mask, weight in penalties.graphs:
        shape = p.axis_sizes
        indices = np.arange(p.n_nodes).reshape(shape)
        for left, right, edge_weight in penalty.graph.indexed_edges:
            a, b = (
                np.take(indices, left, axis=axis).ravel(),
                np.take(indices, right, axis=axis).ravel(),
            )
            cost += (
                weight
                * edge_weight
                * cp.sum_squares(cp.multiply(mask, variable[a] - variable[b]))
                / 2
            )
    objective = cp.Problem(cp.Minimize(cost))
    objective.solve(
        solver=config.solver
    ) if config.solver is not None else objective.solve()
    if (
        objective.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        or variable.value is None
    ):
        raise _NumericalError(
            "reference_failure", f"CVXPY terminated with status {objective.status}"
        )
    theta = np.asarray(variable.value, dtype=np.float64)
    run.iterations = objective.solver_stats.num_iters or 0
    run.assess(theta)
    assert run.objective is not None
    run.history.append(
        IterationRecord(
            iteration=run.iterations,
            objective=run.objective.total,
            stationarity=run.stationarity,
            source_passes=run.evaluation.passes,
            rows_scanned=run.evaluation.rows,
            elapsed_seconds=time.perf_counter() - run.started,
        )
    )
    return theta, None


def solve(
    problem: NumericalProblem,
    *,
    solver: Solver,
    weights: Mapping[str, float],
    cache: SolverCache,
    initial: Array | None = None,
    state: ADMMContinuation | None = None,
) -> NumericalSuccess | NumericalFailure:
    """Solve a validated snapshot; input/data errors propagate, numerical limits do not."""
    penalties = _Penalties(problem, weights)
    effective_initial = initial
    if initial is None and state is not None:
        if state.fingerprint != problem.fingerprint:
            raise ValidationError("Continuation belongs to a different data snapshot")
        effective_initial = state.z
    theta = (
        np.zeros((problem.n_nodes, problem.n_features))
        if effective_initial is None
        else np.array(effective_initial, dtype=np.float64, copy=True)
    )
    if theta.shape != (problem.n_nodes, problem.n_features) or not np.all(
        np.isfinite(theta)
    ):
        raise ValidationError(
            "Initial coefficients must be finite and match the prepared coefficient shape"
        )
    reason = "explicit solver selection"
    if isinstance(solver, AutoSolver):
        if np.any(penalties.l1):
            solver, reason = ADMM(), "nonzero L1 requires a proximal solver"
        elif problem.task == "binary":
            solver, reason = Newton(), "smooth binary logistic objective"
        elif _direct_bytes(problem, penalties) <= solver.direct_max_bytes:
            solver, reason = (
                DirectSolve(max_bytes=solver.direct_max_bytes),
                "uncoupled quadratic blocks fit the configured direct-solve budget"
                if not penalties.has_active_graph
                else "quadratic operator fits the configured direct-solve budget",
            )
        else:
            solver, reason = (
                ConjugateGradient(),
                "quadratic operator exceeds the configured direct-solve budget",
            )
    if isinstance(solver, QuadraticSolver):
        solver = solver.linear_solver
    method = {
        ADMM: "admm",
        Newton: "newton",
        DirectSolve: "direct",
        ConjugateGradient: "cg",
        CVXPYSolver: "cvxpy",
    }[type(solver)]
    run = _Run(
        problem, penalties, cache, method, reason, getattr(solver, "max_passes", None)
    )
    try:
        # Precision is scoped to this execution, never changed at import time.
        with jax.enable_x64(True):
            if isinstance(solver, ADMM):
                theta, continuation = _admm(
                    run, solver, theta, None if initial is not None else state
                )
            elif isinstance(solver, Newton):
                theta, continuation = _newton(run, solver, theta)
            elif isinstance(solver, CVXPYSolver):
                theta, continuation = _cvxpy(run, solver, theta)
            else:
                theta, continuation = _quadratic(
                    run, cast("DirectSolve | ConjugateGradient", solver), theta
                )
        return NumericalSuccess(
            coefficients=theta,
            diagnostics=run.diagnostics(True, "converged"),
            state=continuation,
        )
    except _NumericalError as error:
        return NumericalFailure(
            reason=FailureReason(code=error.code, message=str(error)),
            diagnostics=run.diagnostics(False, error.code),
        )
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        return NumericalFailure(
            reason=FailureReason(code="numerical_breakdown", message=str(error)),
            diagnostics=run.diagnostics(False, "numerical_breakdown"),
        )
