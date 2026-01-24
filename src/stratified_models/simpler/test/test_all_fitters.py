import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stratified_models.simpler.fit import (
    AbstractProblem,
    CVXPYSolver,
    Hyperparameters,
    Solver,
)
from stratified_models.simpler.graph import NetworkXRegularizationGraph
from stratified_models.simpler.loss import SumOfSquaresLoss
from stratified_models.simpler.model import StartifiedModel, Stratification
from stratified_models.simpler.quadratic_fitter import (
    CGSolver,
    DirectSolver,
    QuadraticSolver,
)
from stratified_models.simpler.scalar_function import ScalarFunction, SumOfSquares


def _design_matrix(n: int, m: int) -> np.ndarray:
    x = np.power.outer(np.arange(n, dtype=float), np.arange(m, dtype=float))
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
    ones = np.ones(m)
    for s0 in strat_0.index:
        for s1 in strat_1.index:
            # Two clusters to make Laplacian regularization observable.
            scale = 1.0 if int(s0) == 0 else -3.0
            beta = scale * ones

            df = pd.DataFrame(x_block, columns=regression_features)
            df["strat_0"] = s0
            df["strat_1"] = s1
            frames.append(df)
            y_parts.append(x_block @ beta)

    x = pd.concat(frames, ignore_index=True)
    y = pd.Series(np.concatenate(y_parts), index=x.index)

    num_nodes = graph1.size * graph2.size
    regularizers: dict[str, ScalarFunction] = {"l2": SumOfSquares(shape=(num_nodes, m))}

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
@pytest.mark.parametrize("l2reg", [1e-12, 1e8])
@pytest.mark.parametrize(
    "solver",
    [
        CVXPYSolver(),
        QuadraticSolver(solver=DirectSolver()),
        QuadraticSolver(solver=CGSolver()),
    ],
)
def test_fit(reg1: float, reg2: float, l2reg: float, solver: Solver[object]) -> None:
    m = 2
    problem, hyper = _make_problem_two_graphs(
        reg1=reg1, reg2=reg2, l2_reg=l2reg, m=m, n=3
    )

    model, _compiled = solver.compile_and_solve(problem, hyper)
    assert isinstance(model, StartifiedModel)

    objectives = problem.objectives(model)

    if l2reg >= 1e7:
        assert float(np.linalg.norm(model.theta.to_numpy())) <= 1e-2

    # With very large graph regularization, the corresponding Laplacian energy
    # should be (close to) zero.
    if reg1 >= 1e7:
        assert objectives.laplacians["strat_0"] <= 1e-6
        # Strong Laplacian on `strat_0` should equalize theta across `strat_0`
        # for each fixed `strat_1`.
        for s1 in range(3):
            theta0 = model.theta.loc[(0, s1), :].to_numpy()
            theta1 = model.theta.loc[(1, s1), :].to_numpy()
            assert np.allclose(theta0, theta1, atol=1e-4, rtol=1e-6)
    if reg2 >= 1e7:
        assert objectives.laplacians["strat_1"] <= 1e-6
        # Strong Laplacian on `strat_1` should equalize theta across `strat_1`
        # for each fixed `strat_0`.
        for s0 in range(2):
            theta0 = model.theta.loc[(s0, 0), :].to_numpy()
            theta1 = model.theta.loc[(s0, 1), :].to_numpy()
            theta2 = model.theta.loc[(s0, 2), :].to_numpy()
            assert np.allclose(theta0, theta1, atol=1e-4, rtol=1e-6)
            assert np.allclose(theta1, theta2, atol=1e-4, rtol=1e-6)


@pytest.mark.parametrize(
    "solver",
    [
        CVXPYSolver(),
        QuadraticSolver(solver=DirectSolver()),
        QuadraticSolver(solver=CGSolver()),
    ],
)
def test_ridge_equivalence_when_graph_regs_are_small(solver: Solver[object]) -> None:
    m = 2
    n = 3
    reg1 = 1e-12
    reg2 = 1e-12
    l2reg = 2.5

    problem, hyper = _make_problem_two_graphs(
        reg1=reg1, reg2=reg2, l2_reg=l2reg, m=m, n=n
    )
    model, _compiled = solver.compile_and_solve(problem, hyper)

    # With (effectively) no graph regularization, the solution decouples per-node
    # and matches ridge regression for each group.
    for node, x_slice, y_slice in problem.group_data():
        x = x_slice[problem.regression_features].to_numpy()
        y = y_slice.to_numpy()
        beta_ridge = np.linalg.solve(
            x.T @ x + l2reg * np.eye(m),
            x.T @ y,
        )
        beta_model = model.theta.loc[node, :].to_numpy()
        assert np.allclose(beta_model, beta_ridge, atol=1e-4, rtol=1e-6)
