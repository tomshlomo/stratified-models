from itertools import product
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stratified_models.fitters.admm_fitter import ADMMFitter
from stratified_models.fitters.admm_fitter2 import ADMMFitter2
from stratified_models.fitters.cg_fitter import CGFitter
from stratified_models.fitters.cvxpy_fitter import CVXPYFitter
from stratified_models.fitters.direct_fitter import DirectFitter
from stratified_models.fitters.protocols import (
    Costs,
    NodeData,
    QuadraticStratifiedLinearRegressionProblem,
    StratifiedLinearRegressionFitter,
)
from stratified_models.regularization_graph.cartesian_product import (
    CartesianProductOfGraphs,
)
from stratified_models.regularization_graph.networkx_graph import (
    NetworkXRegularizationGraph,
)


def get_problem(
    reg1: float, reg2: float, l2_reg: float, m: int, n: int
) -> QuadraticStratifiedLinearRegressionProblem:
    graph1 = nx.path_graph(2)
    graph2 = nx.path_graph(3)
    nx.set_edge_attributes(
        graph1, reg1, NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY
    )
    nx.set_edge_attributes(
        graph2, reg2, NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY
    )

    x = (
        np.power.outer(np.arange(n), np.arange(m))
        if m > 1
        else np.arange(n).reshape((-1, 1))
    )
    nodes_data = {
        (1, 2): NodeData(
            x=x,
            y=x @ np.ones(m),
        ),
        (0, 0): NodeData(
            x=x,
            y=x @ np.ones(m) * -3,
        ),
    }
    graph: CartesianProductOfGraphs[int, int] = CartesianProductOfGraphs(
        NetworkXRegularizationGraph(graph1, "1"),
        NetworkXRegularizationGraph(graph2, "2"),
    )
    problem = QuadraticStratifiedLinearRegressionProblem(
        nodes_data=nodes_data,
        graph=graph,
        l2_reg=l2_reg,
        m=m,
    )
    return problem


m_s = [
    2,
    1,
]
params = [
    (1, 1, 1, None),
    (1e6, 1e6, 1e-10, [-1, -1, -1, -1, -1, -1]),
    (1.0, 0.0, 1e-6, [-3, 0, 1, -3, 0, 1]),
    (1e-10, 1e-10, 1e10, [0, 0, 0, 0, 0, 0]),
    (0.0, 1.0, 1e-6, [-3, -3, -3, 1, 1, 1]),
    (1e-10, 1e-10, 1e-6, [-3, 0, 0, 0, 0, 1]),
]
fitters = [
    DirectFitter(),
    ADMMFitter2(),
    CGFitter(),
    CVXPYFitter(),
    ADMMFitter(),
]


@pytest.mark.parametrize(
    ("m", "fitter", "reg1", "reg2", "l2reg", "theta_exp"),
    [(m, fitter, *p) for m, fitter, p in product(m_s, fitters, params)],
)
def test_fit(
    m: int,
    fitter: StratifiedLinearRegressionFitter[Tuple[int, int]],
    reg1: float,
    reg2: float,
    l2reg: float,
    theta_exp: list[float],
) -> None:
    problem = get_problem(reg1=reg1, reg2=reg2, m=m, n=3, l2_reg=l2reg)
    theta, cost = fitter.fit(problem)
    costs_exp = Costs.from_problem_and_theta(problem, theta)
    assert abs(cost - costs_exp.total()) < costs_exp.total() * 1e-3
    if theta_exp:
        theta_exp_df = pd.DataFrame(
            np.tile(theta_exp, (m, 1)),
            columns=problem.graph.nodes,
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
