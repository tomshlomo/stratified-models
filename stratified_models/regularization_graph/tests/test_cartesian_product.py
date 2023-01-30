import itertools
from typing import Iterable, Tuple

import networkx as nx
import numpy as np
import pytest
import scipy

from stratified_models.regularization_graph.cartesian_product import (
    CartesianProductOfGraphs,
)
from stratified_models.regularization_graph.networkx_graph import (
    NetworkXRegularizationGraph,
)
from stratified_models.regularization_graph.regularization_graph import (
    RegularizationGraph,
)
from stratified_models.utils.networkx_utils import cartesian_product


def get_graphs(
    reg: Tuple[float, float, float]
) -> Iterable[tuple[RegularizationGraph[int | tuple[int, ...]], list[nx.Graph]],]:
    graphs = []
    for (name, graph), reg_ in zip(
        [
            ("1", nx.path_graph(2)),
            ("2", nx.cycle_graph(3)),
            ("3", nx.star_graph(4)),
        ],
        reg,
    ):
        nx.set_edge_attributes(
            graph, reg_, NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY
        )
        graphs.append((NetworkXRegularizationGraph(graph, name), graph))
    for n in range(1, len(graphs) + 1):
        for graphs_ in itertools.combinations(graphs, n):
            yield (
                CartesianProductOfGraphs.multi_product(
                    [reg_graph for reg_graph, _ in graphs_]
                ),
                [nx_graph for _, nx_graph in graphs_],
            )


def test_get_laplacian() -> None:
    for graph, nx_graphs in get_graphs((1.0, 2.0, 3.0)):
        lap = graph.laplacian_matrix()
        expected_laplacian = nx.laplacian_matrix(
            cartesian_product(nx_graphs),
            weight=NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY,
        )
        assert not (lap != expected_laplacian).max()


@pytest.mark.parametrize(
    ("reg1", "reg2", "reg3"),
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
    ],
)
def test_laplacian_prox(reg1: float, reg2: float, reg3: float) -> None:
    rho = 1.23
    for graph, nx_graphs in get_graphs((reg1, reg2, reg3)):
        lap = nx.laplacian_matrix(
            cartesian_product(nx_graphs),
            weight=NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY,
        )
        k = lap.shape[0]
        v = np.arange(k * 4, dtype=np.float64).reshape(k, 4)
        expected = scipy.sparse.linalg.spsolve(lap / rho + scipy.sparse.eye(k), v)
        theta = graph.laplacian_prox(v, rho)
        assert np.abs(theta - expected).max() <= 1e-9


@pytest.mark.parametrize(
    ("reg1", "reg2", "reg3"),
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
    ],
)
def test_laplacian_mult(reg1: float, reg2: float, reg3: float) -> None:
    for graph, nx_graphs in get_graphs((reg1, reg2, reg3)):
        lap = nx.laplacian_matrix(
            cartesian_product(nx_graphs),
            weight=NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY,
        )
        k = lap.shape[0]
        v = np.arange(k * 4, dtype=np.float64).reshape(k, 4)
        expected = lap @ v
        out = graph.laplacian_mult(v)
        assert np.abs(out - expected).max() <= 1e-9
