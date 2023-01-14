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


def get_graphs(reg1: float = 1.0, reg2: float = 2.0):
    graph1 = nx.path_graph(2)
    nx.set_edge_attributes(
        graph1, reg1, NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY
    )
    graph2 = nx.cycle_graph(3)
    nx.set_edge_attributes(
        graph2, reg2, NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY
    )
    graph = CartesianProductOfGraphs(
        graph1=NetworkXRegularizationGraph(graph1, "1"),
        graph2=NetworkXRegularizationGraph(graph2, "2"),
    )
    return graph1, graph2, graph


def test_get_laplacian():
    graph1, graph2, graph = get_graphs()
    lap = graph.laplacian_matrix()
    expected_laplacian = nx.laplacian_matrix(
        nx.cartesian_product(graph1, graph2),
        weight=NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY,
    )
    assert (lap == expected_laplacian).min()


@pytest.mark.parametrize(
    ("reg1", "reg2"),
    [
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (1.0, 2.0),
    ],
)
def test_laplacian_prox(reg1: float, reg2: float):
    rho = 1.23
    graph1, graph2, graph = get_graphs(reg1, reg2)
    lap = nx.laplacian_matrix(
        nx.cartesian_product(graph1, graph2),
        weight=NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY,
    )
    v = np.arange(6 * 4).reshape(6, 4)
    # expected_prox_mat = scipy.sparse.linalg.inv(lap / rho + scipy.sparse.eye(6))
    expected = scipy.sparse.linalg.spsolve(lap / rho + scipy.sparse.eye(6), v)
    theta = graph.laplacian_prox(v, rho)
    assert np.abs(theta - expected).max() <= 1e-9
