from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stratified_models.fitters.cvxpy_fitter import CVXPYFitter
from stratified_models.fitters.fitter import (
    CGSolver,
    DirectSolver,
    Fitter,
    QuadraticProblemFitter,
)
from stratified_models.losses import SumOfSquaresLossFactory
from stratified_models.problem import StratifiedLinearRegressionProblem
from stratified_models.regularization_graph.networkx_graph import (
    NetworkXRegularizationGraph,
)
from stratified_models.scalar_function import QuadraticScalarFunction, SumOfSquares


def get_problem(
    reg1: float, reg2: float, l2_reg: float, m: int, n: int
) -> StratifiedLinearRegressionProblem[QuadraticScalarFunction]:
    graph1 = NetworkXRegularizationGraph(nx.path_graph(2), "strat_0")
    graph2 = NetworkXRegularizationGraph(nx.path_graph(3), "strat_1")
    # nx.set_edge_attributes(
    #     graph1, 1.0, NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY
    # )
    # nx.set_edge_attributes(
    #     graph2, 1.0, NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY
    # )

    x = (
        np.power.outer(np.arange(n), np.arange(m))
        if m > 1
        else np.arange(n).reshape((-1, 1))
    )
    y = x @ np.ones(m)

    regression_features = [f"reg_{i}" for i in range(m)]
    df1 = pd.DataFrame(x, columns=regression_features)
    df1["strat_0"] = 1
    df1["strat_1"] = 2
    df2 = pd.DataFrame(x, columns=[f"reg_{i}" for i in range(m)])
    df2["strat_0"] = 0
    df2["strat_1"] = 0

    df = pd.concat([df1, df2], ignore_index=True)
    y = pd.Series(data=np.hstack([y, -3 * y]), index=df.index)

    return StratifiedLinearRegressionProblem(
        x=df,
        y=y,
        loss_factory=SumOfSquaresLossFactory(),
        regularizers=[(SumOfSquares(m), l2_reg)],
        graphs=[(graph1, reg1), (graph2, reg2)],
        regression_features=regression_features,
    )


m_s = [
    1,
    2,
]
params = [
    (1.0, 0.0, 1e-6, [-3, 0, 1, -3, 0, 1]),
    (0.0, 1.0, 1e-6, [-3, -3, -3, 1, 1, 1]),
    (1, 1, 1, None),
    (1e6, 1e6, 1e-10, [-1, -1, -1, -1, -1, -1]),
    (1e-10, 1e-10, 1e10, [0, 0, 0, 0, 0, 0]),
    (1e-10, 1e-10, 1e-6, [-3, 0, 0, 0, 0, 1]),
]
fitters = [
    CVXPYFitter(),
    QuadraticProblemFitter(solver=CGSolver()),
    QuadraticProblemFitter(solver=DirectSolver()),
    # DirectFitter(),
    # ADMMFitter2(),
    # CGFitter(),
    # ADMMFitter(),
]


@pytest.mark.parametrize(
    ("m", "fitter", "reg1", "reg2", "l2reg", "theta_exp"),
    [(m, fitter, *p) for m, fitter, p in product(m_s, fitters, params)],
)
def test_fit(
    m: int,
    fitter: Fitter[QuadraticScalarFunction],
    reg1: float,
    reg2: float,
    l2reg: float,
    theta_exp: list[float],
) -> None:
    problem = get_problem(reg1=reg1, reg2=reg2, m=m, n=3, l2_reg=l2reg)
    theta = fitter.fit(problem)
    # costs_exp = Costs.from_problem_and_theta(problem, theta)
    # assert abs(cost - costs_exp.total()) < costs_exp.total() * 1e-3
    if theta_exp:
        theta_exp_df = pd.DataFrame(
            np.tile(theta_exp, (m, 1)),
            index=problem.regression_features,
            columns=pd.MultiIndex.from_product(
                graph.nodes for graph, _ in problem.graphs
            ),
        ).T
        assert ((theta.df - theta_exp_df).abs() < 1e-3).all().all()


@pytest.mark.parametrize(
    ("reg1", "reg2", "l2reg"),
    [(1, 1, 1), (1, 10, 100), (1, 100, 10)],
)
def test_concensus(
    reg1: float,
    reg2: float,
    l2reg: float,
) -> None:
    problem = get_problem(reg1=reg1, reg2=reg2, m=2, n=3, l2_reg=l2reg)
    theta, cost = fitters[0].fit(problem)
    for fitter in fitters[1:]:
        theta_tmp, cost_tmp = fitter.fit(problem)
        assert (np.abs(theta.df.values - theta_tmp.df.values) < 1e-3).all()
        assert abs(cost - cost_tmp) < 1e-3 * cost
