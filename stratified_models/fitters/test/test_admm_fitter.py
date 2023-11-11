from itertools import product
from math import ceil

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from stratified_models.fitters.admm_fitter import (
    ADMMFitter,
    SeparableProxableScalarFunction,
)
from stratified_models.losses import SumOfSquaresLossFactory
from stratified_models.problem import StratifiedLinearRegressionProblem
from stratified_models.regularization_graph.networkx_graph import (
    NetworkXRegularizationGraph,
)
from stratified_models.scalar_function import L1, SumOfSquares


@pytest.mark.parametrize(
    ("n", "k1", "k2", "max_rank"),
    product(
        [1, 10],
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3, 8, 9, 10, 100],
    ),
)
def test_get_loss(n: int, k1: int, k2: int, max_rank: int) -> None:
    m = 2
    fitter = ADMMFitter(max_rank=max_rank)
    x = np.arange(k1 * k2 * n).astype(float)
    df = (
        pd.DataFrame(
            {
                "x1": x,
                "x2": 1 + x**2,
                "z1": np.mod(x, k1).astype(int),
                "z2": np.mod(x, k2).astype(int),
            }
        )
        .sample(frac=1, random_state=42)
        .reset_index()
    )
    y = df["x1"] * df["z1"] + df["x2"] * df["z2"]
    theta = np.zeros((k1, k2, 2))
    for z1, z2 in product(range(k1), range(k2)):
        theta[z1, z2, 0] = z1
        theta[z1, z2, 1] = z2

    problem = StratifiedLinearRegressionProblem(
        x=df,
        y=y,
        loss_factory=SumOfSquaresLossFactory(),
        regularizers_factories=tuple(),
        graphs=(
            (NetworkXRegularizationGraph(graph=nx.path_graph(k1), name="z1"), 1.0),
            (NetworkXRegularizationGraph(graph=nx.path_graph(k2), name="z2"), 1.0),
        ),
        regression_features=["x1", "x2"],
    )
    loss = fitter._get_loss(problem)
    max_nodes_per_cluster = max(max_rank // m, 1)
    num_of_nodes = df.loc[:, ["z1", "z2"]].drop_duplicates().shape[0]
    expected_num_of_nodes = ceil(num_of_nodes / max_nodes_per_cluster)
    assert len(loss.items) == expected_num_of_nodes
    assert all(
        nodes_indices.shape[0] == max_nodes_per_cluster
        for nodes_indices, _ in loss.items[:-1]
    )
    assert (
        loss.items[-1][0].shape[0] == ((num_of_nodes - 1) % max_nodes_per_cluster) + 1
    )
    val = loss(theta)
    assert val == 0.0


def test_separable_proxable_scalar_function():
    x = np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(float)
    func = SeparableProxableScalarFunction(
        items=[
            (np.array([0, 6]), SumOfSquares(10)),  # [(0,0), (1,2)]
            (np.array([5, 11, 10]), L1()),  # [(1,1), (2,3), (2,2)]
            (np.array([1]), SumOfSquares(5)),  # [(0,1)]
        ]
    )
    out = func(x)
    expected_out = sum(
        float(z.sum())
        for z in [
            x[0, 0] ** 2 / 2,
            x[1, 2] ** 2 / 2,
            x[1, 1],
            x[2, 3],
            x[2, 2],
            x[0, 1] ** 2 / 2,
        ]
    )
    assert out == expected_out

    expected_prox = x.copy()
    expected_prox[0, 0] /= 2.0
    expected_prox[1, 2] /= 2.0
    expected_prox[1, 1] -= 1.0
    expected_prox[2, 3] -= 1.0
    expected_prox[2, 2] -= 1.0
    expected_prox[0, 1] /= 2.0
    prox = func.prox(x, 1)
    assert (prox == expected_prox).all()
