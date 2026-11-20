from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import aslinearoperator

from stratified_models.graph import NetworkXRegularizationGraph
from stratified_models.scalar_function import SparseQuadraticForm


def test_path_graph_builds_weighted_edges_and_laplacian() -> None:
    graph = NetworkXRegularizationGraph.path(3, name="group")

    assert graph.size == 3
    assert graph.name == "group"
    for _, _, data in graph.graph.edges(data=True):
        assert data["weight"] == 1.0

    laplacian = graph.laplacian(axis=0, dims=(3, 2))
    assert isinstance(laplacian, SparseQuadraticForm)
    assert laplacian.axis == 0
    assert laplacian.dims == (3, 2)
    laplacian_operator = aslinearoperator(laplacian.a)
    assert laplacian_operator.shape == (3, 3)


def test_voronoi_graph_has_nodes_and_edges() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph = NetworkXRegularizationGraph.voronoi(points, name="regions")

    assert graph.size == 3
    assert graph.name == "regions"
    assert graph.graph.number_of_edges() > 0
    for _, _, data in graph.graph.edges(data=True):
        assert data["weight"] == 1.0
    for _, data in graph.graph.nodes(data=True):
        assert "point" in data
