import jax
import jax.numpy as jnp
import networkx as nx
import pandas as pd
import pytest

from stratified_models.admm import ConsensusADMM
from stratified_models.graph import NetworkXRegularizationGraph
from stratified_models.loss import SumOfSquaresLoss
from stratified_models.model import Stratification
from stratified_models.scalar_function import ScalarFunction, SumOfSquares
from stratified_models.solvers import (
    AbstractProblem,
    ADMMSolver,
    CGPSDSolver,
    CVXPYSolver,
    DirectPSDSolver,
    Hyperparameters,
    NewtonSolveInfo,
    NewtonSolver,
    SolveInfo,
    Solver,
)

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

ALL_SOLVERS = (
    CVXPYSolver(),
    ADMMSolver(admm=ConsensusADMM(max_iterations=5000)),
    NewtonSolver(),
    NewtonSolver(psd_solver=CGPSDSolver()),
    NewtonSolver.for_quadratic(DirectPSDSolver()),
    NewtonSolver.for_quadratic(CGPSDSolver()),
)


def _design_matrix(n: int, m: int) -> jax.Array:
    a = jnp.arange(n, dtype=float)
    b = jnp.arange(m, dtype=float)
    x = a[:, None] ** b[None, :]
    return x[:, :1] if m == 1 else x


def _make_problem_two_graphs(
    reg1: float,
    reg2: float,
    l2_reg: float,
    m: int,
    n: int,
) -> tuple[AbstractProblem, Hyperparameters]:
    strat_0 = Stratification(index=pd.Index(range(2), name="strat_0"))
    strat_1 = Stratification(index=pd.Index(range(3), name="strat_1"))
    graph1 = NetworkXRegularizationGraph(
        stratification=strat_0, graph=nx.path_graph(strat_0.size)
    )
    graph2 = NetworkXRegularizationGraph(
        stratification=strat_1, graph=nx.path_graph(strat_1.size)
    )

    regression_features = [f"reg_{i}" for i in range(m)]
    x_block = _design_matrix(n=n, m=m)

    frames = []
    y_parts = []
    ones = jnp.ones(m)
    for s0 in strat_0.index:
        for s1 in strat_1.index:
            # Two clusters to make Laplacian regularization observable.
            scale = 1.0 if int(s0) == 0 else -3.0
            beta = scale * ones

            df = pd.DataFrame(jax.device_get(x_block), columns=regression_features)
            df["strat_0"] = s0
            df["strat_1"] = s1
            frames.append(df)
            y_parts.append(x_block @ beta)

    x = pd.concat(frames, ignore_index=True)
    y = pd.Series(jax.device_get(jnp.concatenate(y_parts)), index=x.index)

    regularizers: dict[str, ScalarFunction] = {"l2": SumOfSquares()}

    problem = AbstractProblem(
        x=x,
        y=y,
        regression_features=regression_features,
        graphs=(graph1, graph2),
        loss=SumOfSquaresLoss(),
        regularizers=regularizers,
    )
    hyper = Hyperparameters(
        graphs={str(strat_0.name): reg1, str(strat_1.name): reg2},
        regularizers={"l2": l2_reg},
    )
    return problem, hyper


@pytest.mark.parametrize("reg1", [1e-12, 1e8])
@pytest.mark.parametrize("reg2", [1e-12, 1e8])
@pytest.mark.parametrize("l2reg", [1e3, 1e8])
@pytest.mark.parametrize(
    "solver",
    ALL_SOLVERS,
)
def test_fit(
    reg1: float, reg2: float, l2reg: float, solver: Solver[object, SolveInfo]
) -> None:
    m = 4
    problem, hyper = _make_problem_two_graphs(
        reg1=reg1, reg2=reg2, l2_reg=l2reg, m=m, n=5
    )

    model, info, _compiled = solver.compile_and_solve(problem, hyper)
    if not (
        isinstance(solver, ADMMSolver) and (l2reg >= 1e7 or reg1 >= 1e7 or reg2 >= 1e7)
    ):
        assert info.converged()
    if isinstance(solver, NewtonSolver):
        assert isinstance(info, NewtonSolveInfo)
        assert info.iterations <= 2, (
            "Since the problem is quadratic, the Newton solver should converge "
            "in at most 2 iterations."
        )
    objectives = problem.objectives(model)

    if l2reg >= 1e7:
        assert float(jnp.linalg.norm(jnp.asarray(model.theta))) <= 1e-2

    # With very large graph regularization, the corresponding Laplacian energy
    # should be (close to) zero.
    atol = 1e-3 if isinstance(solver, ADMMSolver) else 2e-4
    if reg1 >= 1e7:
        assert float(objectives.laplacians["strat_0"]) <= 1e-6
        # Strong Laplacian on `strat_0` should equalize theta across `strat_0`
        # for each fixed `strat_1`.
        for s1 in range(3):
            theta0 = model.theta_at_node((0, s1))
            theta1 = model.theta_at_node((1, s1))
            assert bool(
                jnp.allclose(
                    jnp.asarray(theta0), jnp.asarray(theta1), atol=atol, rtol=1e-6
                )
            )
    if reg2 >= 1e7:
        assert float(objectives.laplacians["strat_1"]) <= 1e-6
        # Strong Laplacian on `strat_1` should equalize theta across `strat_1`
        # for each fixed `strat_0`.
        for s0 in range(2):
            theta0 = model.theta_at_node((s0, 0))
            theta1 = model.theta_at_node((s0, 1))
            theta2 = model.theta_at_node((s0, 2))
            assert bool(
                jnp.allclose(
                    jnp.asarray(theta0), jnp.asarray(theta1), atol=atol, rtol=1e-6
                )
            )
            assert bool(
                jnp.allclose(
                    jnp.asarray(theta1), jnp.asarray(theta2), atol=atol, rtol=1e-6
                )
            )


@pytest.mark.parametrize(
    "solver",
    ALL_SOLVERS,
)
def test_ridge_equivalence_when_graph_regs_are_small(
    solver: Solver[object, SolveInfo],
) -> None:
    m = 2
    n = 3
    reg1 = 1e-12
    reg2 = 1e-12
    l2reg = 2.5

    problem, hyper = _make_problem_two_graphs(
        reg1=reg1, reg2=reg2, l2_reg=l2reg, m=m, n=n
    )
    model, info, _compiled = solver.compile_and_solve(problem, hyper)
    if not (
        isinstance(solver, ADMMSolver) and (l2reg >= 1e7 or reg1 >= 1e7 or reg2 >= 1e7)
    ):
        assert info.converged()

    # With (effectively) no graph regularization, the solution decouples per-node
    # and matches ridge regression for each group.
    atol = 1e-3 if isinstance(solver, ADMMSolver) else 1e-4
    for node, x_slice, y_slice in problem.group_data():
        x = jnp.asarray(x_slice[problem.regression_features].to_numpy())
        y = jnp.asarray(y_slice.to_numpy())
        beta_ridge = jnp.linalg.solve(
            x.T @ x + l2reg * jnp.eye(m),
            x.T @ y,
        )
        beta_model = model.theta_at_node(node)
        assert bool(
            jnp.allclose(jnp.asarray(beta_model), beta_ridge, atol=atol, rtol=1e-6)
        )


def test_all_solvers_achieve_same_objective_value() -> None:
    problem, hyper = _make_problem_two_graphs(
        reg1=1.0,
        reg2=2.0,
        l2_reg=0.3,
        m=4,
        n=5,
    )

    totals: list[float] = []
    for solver in ALL_SOLVERS:
        model, info, _compiled = solver.compile_and_solve(problem, hyper)
        assert info.converged()
        total = problem.objectives(model).total(hyper)
        totals.append(float(total))

    baseline = totals[0]
    for i, value in enumerate(totals[1:]):
        # ADMM (index 0 in totals[1:] since it is second in ALL_SOLVERS)
        # might be less precise
        tol = 1e-4 if i == 0 else 1e-5
        assert abs(value - baseline) <= tol * max(1.0, abs(baseline))
