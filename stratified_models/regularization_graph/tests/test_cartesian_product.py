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


def test_all():
    graph1 = nx.path_graph(2)
    graph2 = nx.path_graph(3)
    nx.set_edge_attributes(graph1, 1, NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY)
    nx.set_edge_attributes(
        graph2, 100, NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY
    )
    graph: CartesianProductOfGraphs[int, int] = CartesianProductOfGraphs(
        NetworkXRegularizationGraph(graph1, "1"),
        NetworkXRegularizationGraph(graph2, "2"),
    )
    nx_graph = cartesian_product([graph1, graph2])
    lap = graph.laplacian_matrix()
    lap_exp = nx.laplacian_matrix(
        nx_graph, weight=NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY
    )
    diff = (lap - lap_exp).toarray()
    assert np.all(diff == 0)

    rng = np.random.RandomState(42)
    x = rng.standard_normal((6, 2))
    y = graph.laplacian_mult(x)
    y_exp = lap_exp @ x
    diff = y - y_exp
    assert np.all(np.abs(diff) < 1e-9)

    y = graph.laplacian_quad_form(x)
    y_exp = np.trace(x.T @ lap_exp @ x)
    assert y == y_exp

    rho = 1.23
    y = graph.laplacian_prox(x, rho)
    y_exp = scipy.sparse.linalg.spsolve(lap_exp * (2 / rho) + scipy.sparse.eye(6), x)
    diff = y - y_exp
    assert np.all(np.abs(diff) < 1e-9)


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
        expected = scipy.sparse.linalg.spsolve(lap * (2 / rho) + scipy.sparse.eye(k), v)
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
