"""Independent optimality and replay contracts for production solvers."""

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pytest

from stratified_models.data import NumericBatch
from stratified_models.solvers import (
    ADMM,
    AutoSolver,
    ConjugateGradient,
    CVXPYSolver,
    DirectSolve,
    NewtonSolver,
    NumericalFailure,
    NumericalSuccess,
    QuadraticStatistics,
    Solver,
    SolverCache,
    solve,
)
from stratified_models.specs import L1, L2, Axis, Graph, Laplacian


@dataclass(frozen=True)
class _Problem:
    task: Literal["regression", "binary"]
    x: np.ndarray
    y: np.ndarray
    nodes: np.ndarray
    axes: tuple
    penalties: tuple
    statistics: QuadraticStatistics | None
    batch_size: int = 7
    fingerprint: str = "data-one"
    design_fingerprint: str = "design-one"
    features: tuple = ("intercept", "x")

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def axis_sizes(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.axis_sizes))

    @property
    def n_features(self) -> int:
        return len(self.features)

    def iter_numeric(self) -> Iterator[NumericBatch]:
        for first in range(0, self.n_samples, self.batch_size):
            last = first + self.batch_size
            yield NumericBatch(
                x=self.x[first:last],
                y=self.y[first:last],
                nodes=self.nodes[first:last],
                row_ids=np.arange(first, min(last, self.n_samples)),
            )


def _problem(
    task: Literal["regression", "binary"] = "regression",
    *,
    statistics: bool = True,
    l1: float = 0.0,
    batch_size: int = 7,
) -> _Problem:
    rng = np.random.default_rng(42)
    axis = Axis(name="region", nodes=("right", "left"))
    x = np.column_stack((np.ones(60), rng.normal(size=60)))
    nodes = np.tile(np.arange(2), 30)
    coefficients = np.array([[0.6, 1.8], [-0.5, -1.2]])
    prediction = np.sum(x * coefficients[nodes], axis=1)
    y = (
        prediction + rng.normal(scale=0.1, size=60)
        if task == "regression"
        else rng.binomial(1, 1 / (1 + np.exp(-prediction))).astype(float)
    )
    penalties = (
        L1(name="sparse", weight=l1, features=("x",)),
        L2(name="ridge", weight=0.05, features=("x",)),
        Laplacian(
            name="space",
            weight=0.08,
            features=("intercept", "x"),
            graph=Graph.path(axis=axis),
        ),
    )
    stats = None
    if statistics and task == "regression":
        xtx = np.zeros((2, 2, 2))
        xty = np.zeros((2, 2))
        np.add.at(xtx, nodes, x[:, :, None] * x[:, None, :])
        np.add.at(xty, nodes, x * y[:, None])
        stats = QuadraticStatistics(xtx=xtx, xty=xty, yty=y @ y)
    return _Problem(task, x, y, nodes, (axis,), penalties, stats, batch_size)


@pytest.mark.parametrize("task", ["regression", "binary"])
@pytest.mark.parametrize("l1", [0.0, 0.1])
def test_admm_matches_independent_convex_reference(
    task: Literal["regression", "binary"], l1: float
) -> None:
    pytest.importorskip("cvxpy")
    problem = _problem(task, l1=l1)
    reference = solve(problem, solver=CVXPYSolver(), weights={}, cache=SolverCache())
    result = solve(
        problem,
        solver=ADMM(max_iterations=1500, atol=1e-6, rtol=1e-5),
        weights={},
        cache=SolverCache(),
    )
    assert isinstance(reference, NumericalSuccess)
    assert isinstance(result, NumericalSuccess), result
    assert result.diagnostics.objective.total == pytest.approx(
        reference.diagnostics.objective.total, abs=2e-6
    )
    assert np.allclose(result.coefficients, reference.coefficients, atol=3e-3)
    record = result.diagnostics.history[-1]
    assert record.primal_residual <= record.primal_threshold
    assert record.dual_residual <= record.dual_threshold
    assert not result.coefficients.flags.writeable


@pytest.mark.parametrize(
    "strategy", [DirectSolve(), ConjugateGradient(), NewtonSolver(), AutoSolver()]
)
def test_smooth_regression_strategies_match_reference(strategy: Solver) -> None:
    problem = _problem()
    reference = solve(problem, solver=DirectSolve(), weights={}, cache=SolverCache())
    result = solve(problem, solver=strategy, weights={}, cache=SolverCache())
    assert isinstance(result, NumericalSuccess), result
    assert isinstance(reference, NumericalSuccess)
    assert np.allclose(result.coefficients, reference.coefficients, atol=1e-6)


def test_streamed_squared_loss_and_batch_partitions_match_statistics() -> None:
    reference = solve(_problem(), solver=DirectSolve(), weights={}, cache=SolverCache())
    assert isinstance(reference, NumericalSuccess)
    for batch_size in (1, 13, 60):
        result = solve(
            _problem(statistics=False, batch_size=batch_size),
            solver=ConjugateGradient(),
            weights={},
            cache=SolverCache(),
        )
        assert isinstance(result, NumericalSuccess), result
        assert np.allclose(result.coefficients, reference.coefficients, atol=1e-7)
        assert result.diagnostics.rows_scanned == result.diagnostics.source_passes * 60


def test_exact_zero_admm_solution_converges_immediately_without_penalty_explosion() -> (
    None
):
    base = _problem()
    zero = _Problem(
        "regression",
        base.x,
        np.zeros(60),
        base.nodes,
        base.axes,
        base.penalties,
        QuadraticStatistics(xtx=base.statistics.xtx, xty=np.zeros((2, 2)), yty=0.0),
    )
    result = solve(zero, solver=ADMM(), weights={}, cache=SolverCache())
    assert isinstance(result, NumericalSuccess)
    assert result.diagnostics.iterations == 1
    assert np.array_equal(result.coefficients, np.zeros((2, 2)))


def test_cache_dependencies_and_explicit_continuation() -> None:
    problem = _problem()
    cache = SolverCache()
    first = solve(problem, solver=DirectSolve(), weights={}, cache=cache)
    second = solve(problem, solver=DirectSolve(), weights={}, cache=cache)
    changed = solve(problem, solver=DirectSolve(), weights={"ridge": 0.2}, cache=cache)
    assert all(
        isinstance(value, NumericalSuccess) for value in (first, second, changed)
    )
    assert second.diagnostics.cache.hits == 1
    assert second.diagnostics.cache.factorizations == 0
    assert changed.diagnostics.cache.factorizations == 1
    initial = solve(problem, solver=ADMM(), weights={}, cache=cache)
    assert isinstance(initial, NumericalSuccess)
    continued = solve(
        problem, solver=ADMM(), weights={}, cache=cache, state=initial.state
    )
    assert isinstance(continued, NumericalSuccess)
    assert continued.diagnostics.iterations <= initial.diagnostics.iterations


def test_budget_failures_are_typed_and_auto_records_choice() -> None:
    result = solve(
        _problem(), solver=DirectSolve(max_bytes=1), weights={}, cache=SolverCache()
    )
    assert isinstance(result, NumericalFailure)
    assert result.reason.code == "memory_limit"
    result = solve(
        _problem("binary"),
        solver=NewtonSolver(max_passes=1),
        weights={},
        cache=SolverCache(),
    )
    assert isinstance(result, NumericalFailure)
    assert result.reason.code == "pass_limit"
    result = solve(
        _problem(l1=0.1), solver=AutoSolver(), weights={}, cache=SolverCache()
    )
    assert isinstance(result, NumericalSuccess)
    assert result.diagnostics.solver == "admm"
    assert "L1" in result.diagnostics.selection_reason


def test_cartesian_graph_axis_order_and_feature_masks_match_reference() -> None:
    pytest.importorskip("cvxpy")
    rng = np.random.default_rng(4)
    first = Axis(name="first", nodes=("b", "a"))
    second = Axis(name="second", nodes=(2, 0, 1))
    nodes = np.repeat(np.arange(6), 5)
    x = np.column_stack((np.ones(30), rng.normal(size=30)))
    y = rng.normal(size=30)
    penalties = (
        L2(name="ridge", weight=0.1, features=("intercept", "x")),
        Laplacian(
            name="one", weight=0.3, features=("x",), graph=Graph.path(axis=first)
        ),
        Laplacian(
            name="two",
            weight=0.8,
            features=("intercept",),
            graph=Graph.path(axis=second),
        ),
    )
    problem = _Problem("regression", x, y, nodes, (first, second), penalties, None)
    reference = solve(problem, solver=CVXPYSolver(), weights={}, cache=SolverCache())
    result = solve(problem, solver=DirectSolve(), weights={}, cache=SolverCache())
    assert isinstance(result, NumericalSuccess)
    assert isinstance(reference, NumericalSuccess)
    assert np.allclose(result.coefficients, reference.coefficients, atol=1e-6)
    assert result.diagnostics.objective.total == pytest.approx(
        reference.diagnostics.objective.total, abs=1e-8
    )


def test_changed_targets_reuse_design_factor_without_reusing_old_right_hand_side() -> (
    None
):
    problem = _problem()
    assert problem.statistics is not None
    shifted = replace(
        problem,
        y=2 * problem.y,
        fingerprint="changed-targets",
        statistics=QuadraticStatistics(
            xtx=problem.statistics.xtx,
            xty=2 * problem.statistics.xty,
            yty=4 * problem.statistics.yty,
        ),
    )
    cache = SolverCache()
    first = solve(problem, solver=DirectSolve(), weights={}, cache=cache)
    second = solve(shifted, solver=DirectSolve(), weights={}, cache=cache)
    assert isinstance(first, NumericalSuccess)
    assert isinstance(second, NumericalSuccess)
    assert second.diagnostics.cache.hits == 1
    assert second.diagnostics.cache.factorizations == 0
    assert np.allclose(second.coefficients, 2 * first.coefficients)
    with pytest.raises(ValueError):
        first.coefficients.setflags(write=True)


def test_extreme_logistic_logits_remain_finite_and_precision_is_execution_scoped() -> (
    None
):
    import jax

    problem = _problem("binary")
    initial = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])
    before = jax.config.x64_enabled
    result = solve(
        problem, solver=NewtonSolver(), weights={}, cache=SolverCache(), initial=initial
    )
    assert isinstance(result, NumericalSuccess), result
    assert np.isfinite(result.coefficients).all()
    assert math.isfinite(result.diagnostics.objective.total)
    assert jax.config.x64_enabled == before


def test_auto_uses_small_uncoupled_blocks_for_many_strata() -> None:
    axis = Axis(name="region", nodes=tuple(range(2_500)))
    nodes = np.repeat(np.arange(100), 4)
    x = np.tile(np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]), (100, 1))
    y = 1 + 2 * x[:, 1]
    xtx, xty = np.zeros((2_500, 2, 2)), np.zeros((2_500, 2))
    np.add.at(xtx, nodes, x[:, :, None] * x[:, None, :])
    np.add.at(xty, nodes, x * y[:, None])
    statistics = QuadraticStatistics(xtx=xtx, xty=xty, yty=y @ y)
    problem = _Problem("regression", x, y, nodes, (axis,), (), statistics)
    # A dense coefficient Hessian would take hundreds of MB; blocks fit below 1 MB.
    result = solve(
        problem,
        solver=AutoSolver(direct_max_bytes=1_000_000),
        weights={},
        cache=SolverCache(),
    )
    assert isinstance(result, NumericalSuccess), result
    assert result.diagnostics.solver == "direct"
    assert "uncoupled" in result.diagnostics.selection_reason
    assert np.allclose(result.coefficients[:100], [1.0, 2.0])
    assert np.array_equal(result.coefficients[100:], np.zeros((2_400, 2)))


def test_uncoupled_direct_selects_minimum_norm_for_singular_and_unobserved_strata() -> (
    None
):
    axis = Axis(name="region", nodes=(0, 1, 2))
    x = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y, nodes = np.array([2.0, 2.0, 3.0]), np.array([0, 0, 2])
    problem = _Problem("regression", x, y, nodes, (axis,), (), None)
    result = solve(problem, solver=DirectSolve(), weights={}, cache=SolverCache())
    assert isinstance(result, NumericalSuccess), result
    assert np.allclose(result.coefficients, [[1, 1], [0, 0], [0, 0]])
    assert result.diagnostics.objective.loss == pytest.approx(1.5)
    assert result.diagnostics.stationarity < 1e-12


def test_coupled_direct_cache_hit_avoids_dense_matrix_reassembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scipy import linalg

    problem, cache = _problem(), SolverCache()
    first = solve(problem, solver=DirectSolve(), weights={}, cache=cache)
    assert isinstance(first, NumericalSuccess)

    def unexpected_reassembly(*args: object, **kwargs: object) -> None:
        raise AssertionError("A factor cache hit must not rebuild the dense Hessian")

    monkeypatch.setattr(linalg, "block_diag", unexpected_reassembly)
    second = solve(problem, solver=DirectSolve(), weights={}, cache=cache)
    assert isinstance(second, NumericalSuccess)
    assert second.diagnostics.cache.hits == 1
    assert np.allclose(first.coefficients, second.coefficients)
