from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stratified_models.fitters.admm_fitter import ADMMFitter
from stratified_models.fitters.cvxpy_fitter import CVXPYFitter
from stratified_models.fitters.fitter import Fitter, RefitDataType
from stratified_models.losses import LogisticLossFactory
from stratified_models.problem import StratifiedLinearRegressionProblem
from stratified_models.regularization_graph.networkx_graph import (
    NetworkXRegularizationGraph,
)
from stratified_models.regularizers import SumOfSquaresRegularizerFactory
from stratified_models.scalar_function import (
    Array,
    QuadraticScalarFunction,
    ScalarFunction,
)


def get_problem(
    reg1: float,
    reg2: float,
    l2_reg: float,
    n: int,
) -> StratifiedLinearRegressionProblem[QuadraticScalarFunction[Array]]:
    """
    both       +---           +---       ][      +++-            +++-
    group12    +---               ][             +++-
    group00                   +---               ][              +++-
              |    |    |    |    |    |    |    |    |    |    |    |
              -6   -5   -4   -3   -2   -1   0    1    2    3    4    5
    """
    graph1 = NetworkXRegularizationGraph(nx.path_graph(2), "strat_0")
    graph2 = NetworkXRegularizationGraph(nx.path_graph(3), "strat_1")
    dfs = []
    for x_left, y, start_0, start_1 in [
        (-3, -1, 0, 0),
        (4, 1, 0, 0),
        (-6, -1, 1, 2),
        (1, 1, 1, 2),
    ]:
        df = pd.DataFrame()
        df["x"] = np.linspace(x_left, x_left + 1, n)
        df["y"] = y
        if y > 0:
            df.at[n - 1, "y"] = -1
        else:
            df.at[0, "y"] = 1
        df["strat_0"] = start_0
        df["strat_1"] = start_1
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["one"] = 1.0

    return StratifiedLinearRegressionProblem(
        df=df,
        target_column="y",
        loss_factory=LogisticLossFactory(),
        regularizers_factories=((SumOfSquaresRegularizerFactory(), l2_reg),),
        graphs=((graph1, reg1), (graph2, reg2)),
        regression_features=["x", "one"],
    )


params = [
    (1e-10, 1e-10, 1e-3, [1, 0, 0, 0, 0, -2]),
    (1e0, 1e-6, 1e-3, [1, 0, -2, 1, 0, -2]),
    (1e-6, 1e0, 1e-3, [1, 1, 1, -2, -2, -2]),
    (1e3, 1e3, 1e-3, [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]),
    (1e-10, 1e-10, 1e6, [0] * 6),
    (1e0, 1e-6, 1e6, [0] * 6),
    (1e-6, 1e0, 1e6, [0] * 6),
    (1e3, 1e3, 1e6, [0] * 6),
]
fitters = [
    ADMMFitter(),
    CVXPYFitter(),
]


@pytest.mark.parametrize(
    ("fitter", "reg1", "reg2", "l2_reg", "expected_intercepts"),
    [(fitter, *p) for fitter, p in product(fitters, params)],
)
def test_fit(
    fitter: Fitter[ScalarFunction[Array], RefitDataType],
    reg1: float,
    reg2: float,
    l2_reg: float,
    expected_intercepts: list[float],
) -> None:
    problem = get_problem(reg1=reg1, reg2=reg2, n=3, l2_reg=l2_reg)
    theta, _, cost = fitter.fit(problem)
    cost_exp1 = problem.cost(theta)
    assert abs(cost - cost_exp1) <= 1e-3 * cost_exp1 + 1e-6
    df_plot = pd.DataFrame()
    df_plot["x"] = np.linspace(-10, 10, 1000)
    df_plot["one"] = 1.0
    for z, theta_loc in theta.df.iterrows():
        df_plot[f"y_{z}"] = 1 / (
            1 + np.exp(-(df_plot["x"] * theta_loc["x"] + theta_loc["one"]))
        )
    # import plotly.express as px
    # import plotly.io
    # plotly.io.renderers.default = "browser"
    # px.line(
    #     df_plot,
    #     x='x',
    #     y=list(df_plot.columns[2:]),
    # ).show()
    intercepts = -theta.df["one"] / theta.df["x"]
    expected_intercepts = np.array(expected_intercepts)
    tol = 1e-6 + 1e-3 * theta.df["one"].abs() / theta.df["x"] ** 2
    assert all(np.abs(intercepts - expected_intercepts) <= tol)
