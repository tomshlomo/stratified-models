from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stratified_models.fitters.admm_fitter import ADMMFitter
from stratified_models.fitters.cvxpy_fitter import CVXPYFitter
from stratified_models.fitters.fitter import Fitter, ProblemUpdate, RefitDataType
from stratified_models.fitters.quadratic_fitter import (
    CGSolver,
    DirectSolver,
    QuadraticProblemFitter,
)
from stratified_models.losses import SumOfSquaresLossFactory
from stratified_models.problem import StratifiedLinearRegressionProblem, Theta
from stratified_models.regularization_graph.networkx_graph import (
    NetworkXRegularizationGraph,
)
from stratified_models.regularizers import SumOfSquaresRegularizerFactory
from stratified_models.scalar_function import Array, QuadraticScalarFunction


def get_problem(
    reg1: float, reg2: float, l2_reg: float, m: int, n: int
) -> StratifiedLinearRegressionProblem[QuadraticScalarFunction[Array]]:
    graph1 = NetworkXRegularizationGraph(nx.path_graph(2), "strat_0")
    graph2 = NetworkXRegularizationGraph(nx.path_graph(3), "strat_1")
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
        regularizers_factories=((SumOfSquaresRegularizerFactory(), l2_reg),),
        graphs=((graph1, reg1), (graph2, reg2)),
        regression_features=regression_features,
    )


m_s = [
    2,
    1,
]
params = [
    (1e-10, 1e-10, 1e-3, [-3, 0, 0, 0, 0, 1]),
    (1, 1, 1, None),
    (1.0, 1e-10, 1e-10, [-3, 0, 1, -3, 0, 1]),
    (1e6, 1e6, 1e-10, [-1, -1, -1, -1, -1, -1]),
    (1e-6, 1.0, 1e-6, [-3, -3, -3, 1, 1, 1]),
    (1e-10, 1e-10, 1e10, [0, 0, 0, 0, 0, 0]),
]
fitters = [
    ADMMFitter(),
    QuadraticProblemFitter(solver=DirectSolver()),
    CVXPYFitter(),
    QuadraticProblemFitter(solver=CGSolver()),
]


@pytest.mark.parametrize(
    ("m", "fitter", "reg1", "reg2", "l2reg", "theta_exp"),
    [(m, fitter, *p) for m, fitter, p in product(m_s, fitters, params)],
)
def test_fit(
    m: int,
    fitter: Fitter[QuadraticScalarFunction[Array], RefitDataType],
    reg1: float,
    reg2: float,
    l2reg: float,
    theta_exp: list[float],
) -> None:
    problem = get_problem(reg1=reg1, reg2=reg2, m=m, n=3, l2_reg=l2reg)
    theta, _, cost = fitter.fit(problem)
    cost_exp1 = problem.cost(theta)
    assert abs(cost - cost_exp1) <= 1e-3 * cost_exp1 + 1e-6
    if theta_exp:
        theta_exp_df = pd.DataFrame(
            np.tile(theta_exp, (m, 1)),
            index=problem.regression_features,
            columns=pd.MultiIndex.from_product(
                graph.nodes for graph, _ in problem.graphs
            ),
        ).T
        cost_exp2 = problem.cost(Theta(theta_exp_df, shape=problem.theta_shape()))
        assert (abs(cost - cost_exp2) <= 1e-3 * cost_exp2 + 1e-6) or (
            (theta.df - theta_exp_df).abs() < 1e-3
        ).all().all()


@pytest.mark.parametrize(
    ("m", "fitter", "reg1", "reg2", "l2reg", "theta_exp"),
    [(m, fitter, *p) for m, fitter, p in product(m_s, fitters, params)],
)
def test_refit(
    m: int,
    fitter: Fitter[QuadraticScalarFunction[Array], RefitDataType],
    reg1: float,
    reg2: float,
    l2reg: float,
    theta_exp: list[float],
) -> None:
    problem = get_problem(reg1=1, reg2=1, m=m, n=3, l2_reg=1)
    _, refit_data, _ = fitter.fit(problem)

    problem_update = ProblemUpdate(
        new_regularization_gammas=[l2reg],
        new_graph_gammas=[reg1, reg2],
    )
    theta, _, cost = fitter.refit(problem_update=problem_update, refit_data=refit_data)
    new_problem = problem_update.apply(problem)
    cost_exp1 = new_problem.cost(theta)
    assert abs(cost - cost_exp1) <= 1e-3 * cost_exp1 + 1e-6

    if theta_exp:
        theta_exp_df = pd.DataFrame(
            np.tile(theta_exp, (m, 1)),
            index=problem.regression_features,
            columns=pd.MultiIndex.from_product(
                graph.nodes for graph, _ in problem.graphs
            ),
        ).T
        cost_exp2 = new_problem.cost(Theta(theta_exp_df, shape=problem.theta_shape()))
        assert (abs(cost - cost_exp2) <= 1e-3 * cost_exp2 + 1e-6) or (
            (theta.df - theta_exp_df).abs() < 1e-3
        ).all().all()


def get_problem_single_graph(
    laplace_reg: float,
    l2_reg: float,
    m: int,
    n: int,
) -> StratifiedLinearRegressionProblem[QuadraticScalarFunction[Array]]:
    graph = NetworkXRegularizationGraph(nx.path_graph(3), "z")
    x = (
        np.power.outer(np.arange(n), np.arange(m))
        if m > 1
        else np.arange(n).reshape((-1, 1))
    )
    y = x @ np.ones(m)

    regression_features = [f"reg_{i}" for i in range(m)]
    df1 = pd.DataFrame(x, columns=regression_features)
    df1["z"] = 0
    df2 = pd.DataFrame(x, columns=[f"reg_{i}" for i in range(m)])
    df2["z"] = 2

    df = pd.concat([df1, df2], ignore_index=True)
    y = pd.Series(data=np.hstack([y, -3 * y]), index=df.index)

    return StratifiedLinearRegressionProblem(
        x=df,
        y=y,
        loss_factory=SumOfSquaresLossFactory(),
        regularizers_factories=((SumOfSquaresRegularizerFactory(), l2_reg),),
        graphs=((graph, laplace_reg),),
        regression_features=regression_features,
    )


params_single_graph = [
    (1e-6, 1e-3, [1, 0, -3]),
    (1e6, 1e-3, [-1, -1, -1]),
    (1e-3, 1e6, [0, 0, 0]),
]


@pytest.mark.parametrize(
    ("m", "fitter", "laplace_reg", "l2reg", "theta_exp"),
    [(m, fitter, *p) for m, fitter, p in product(m_s, fitters, params_single_graph)],
)
def test_fit_single_graph(
    m: int,
    fitter: Fitter[QuadraticScalarFunction[Array], RefitDataType],
    laplace_reg: float,
    l2reg: float,
    theta_exp: list[float],
) -> None:
    problem = get_problem_single_graph(laplace_reg=laplace_reg, m=m, n=3, l2_reg=l2reg)
    theta, _, cost = fitter.fit(problem)
    cost_exp1 = problem.cost(theta)
    assert abs(cost - cost_exp1) <= 1e-3 * cost_exp1 + 1e-6
    if theta_exp:
        theta_exp_df = pd.DataFrame(
            np.tile(theta_exp, (m, 1)),
            index=problem.regression_features,
        ).T
        theta_exp_df.index.name = "z"
        cost_exp2 = problem.cost(Theta(theta_exp_df, shape=problem.theta_shape()))
        assert (abs(cost - cost_exp2) <= 1e-3 * cost_exp2 + 1e-6) or (
            (theta.df - theta_exp_df).abs() < 1e-3
        ).all().all()
