from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stratified_models.fitters.cg_fitter import CGFitter
from stratified_models.fitters.cvxpy_fitter import CVXPYFitter
from stratified_models.fitters.direct_fitter import DirectFitter
from stratified_models.fitters.protocols import (
    LAPLACE_REG_PARAM_KEY,
    NodeData,
    StratifiedLinearRegressionFitter,
)


def get_data(reg1: float, reg2: float, m: int, n: int):
    graphs = {
        "1": nx.path_graph(2),
        "2": nx.path_graph(3),
    }
    nx.set_edge_attributes(graphs["1"], reg1, LAPLACE_REG_PARAM_KEY)
    nx.set_edge_attributes(graphs["2"], reg2, LAPLACE_REG_PARAM_KEY)

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
    return graphs, nodes_data


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
    fitter: StratifiedLinearRegressionFitter,
    reg1: float,
    reg2: float,
    l2reg: float,
    theta_exp: list[float],
):
    graphs, nodes_data = get_data(reg1=reg1, reg2=reg2, m=m, n=3)
    theta = fitter.fit(nodes_data=nodes_data, graphs=graphs, l2_reg=l2reg, m=m)
    theta_exp_df = pd.DataFrame(
        np.tile(theta_exp, (m, 1)),
        columns=pd.MultiIndex.from_product([range(2), range(3)], names=graphs.keys()),
    ).T
    assert ((theta - theta_exp_df).abs() < 1e-3).all().all()
