from itertools import product
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stratified_models.fitters.admm_fitter import ADMMFitter
from stratified_models.fitters.cg_fitter import CGFitter
from stratified_models.fitters.cvxpy_fitter import CVXPYFitter
from stratified_models.fitters.direct_fitter import DirectFitter
from stratified_models.fitters.protocols import (
    NodeData,
    StratifiedLinearRegressionFitter,
)
from stratified_models.regularization_graph.cartesian_product import (
    CartesianProductOfGraphs,
)
from stratified_models.regularization_graph.networkx_graph import (
    NetworkXRegularizationGraph,
)
from stratified_models.regularization_graph.regularization_graph import Name


def get_data(
    reg1: float, reg2: float, m: int, n: int
) -> Tuple[nx.Graph, nx.Graph, NodeData]:
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
    return graph1, graph2, nodes_data


m_s = [
    2,
    1,
]
params = [
    (1.0, 0.0, 1e-6, [-3, 0, 1, -3, 0, 1]),
    (1e-10, 1e-10, 1e10, [0, 0, 0, 0, 0, 0]),
    (1e6, 1e6, 1e-10, [-1, -1, -1, -1, -1, -1]),
    (0.0, 1.0, 1e-6, [-3, -3, -3, 1, 1, 1]),
    (1e-10, 1e-10, 1e-6, [-3, 0, 0, 0, 0, 1]),
]
fitters = [
    ADMMFitter(),
    CGFitter(),
    DirectFitter(),
    CVXPYFitter(),
]


@pytest.mark.parametrize(
    ("m", "fitter", "reg1", "reg2", "l2reg", "theta_exp"),
    [(m, fitter, *p) for m, fitter, p in product(m_s, fitters, params)],
)
def test_fit(
    m: int,
    fitter: StratifiedLinearRegressionFitter[Tuple[int, int], Name],
    reg1: float,
    reg2: float,
    l2reg: float,
    theta_exp: list[float],
) -> None:
    graph1, graph2, nodes_data = get_data(reg1=reg1, reg2=reg2, m=m, n=3)
    graph = CartesianProductOfGraphs(
        NetworkXRegularizationGraph(graph1, "1"),
        NetworkXRegularizationGraph(graph2, "2"),
    )
    # graph = NetworkXRegularizationGraph(
    #     graph=cartesian_product(graphs.values()), names=graphs.keys()
    # )
    theta = fitter.fit(nodes_data=nodes_data, graph=graph, l2_reg=l2reg, m=m)
    theta_exp_df = pd.DataFrame(np.tile(theta_exp, (m, 1)), columns=graph.nodes()).T
    assert ((theta - theta_exp_df).abs() < 1e-3).all().all()
