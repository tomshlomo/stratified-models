from __future__ import annotations

from typing import Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import scipy
from numpy import typing as npt

from stratified_models.regularization_graph.regularization_graph import (
    Node,
    RegularizationGraph,
)

Node1 = TypeVar("Node1")
Node2 = TypeVar("Node2")

NestedNode = Union[Node, Tuple["NestedNode", Node]]


def kron_sum(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.kron(np.eye(b.shape[0]), a) + np.kron(b, np.eye(a.shape[0]))


class CartesianProductOfGraphs(RegularizationGraph[Tuple[Node1, Node2]]):
    def __init__(
        self,
        graph1: RegularizationGraph[Node1],
        graph2: RegularizationGraph[Node2],
    ) -> None:
        nodes = pd.MultiIndex.from_product([graph1.nodes, graph2.nodes])
        nodes = nodes.to_flat_index()
        nodes.name = (graph1.name(), graph2.name())
        super().__init__(nodes=nodes)
        self.graph1 = graph1
        self.graph2 = graph2

    def laplacian_matrix(self) -> scipy.sparse.spmatrix:
        lap1 = self.graph1.laplacian_matrix()
        lap2 = self.graph2.laplacian_matrix()
        return scipy.sparse.kronsum(lap2, lap1)

    # def laplacian_prox_matrix(self, rho: float) -> npt.NDArray[np.float64]:
    #     out = scipy.sparse.linalg.inv(
    #         self.laplacian_matrix()
    #         + (2 / rho) * scipy.sparse.eye(self.number_of_nodes())
    #     )
    #     p1 = self.graph1.laplacian_prox_matrix(rho * 2)
    #     p2 = self.graph2.laplacian_prox_matrix(rho * 2)
    #     p = scipy.sparse.kronsum(p2, p1)
    #     # p = kron_sum(p2, p1)
    #     return out

    def laplacian_mult(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        (A*B is kron(A, B))
        L x = (I * L2 + L1 * I) x
        = (I * L2) x + (L1 * I) x
        = vec( mat(x) L2 ) + vec( L1 mat(x) )
        """
        m = x.shape[-1]
        k1 = self.graph1.number_of_nodes()
        k2 = self.graph2.number_of_nodes()
        x = x.reshape((k1, k2, m))
        l1x = self.graph1.laplacian_mult(x.reshape((k1, -1))).reshape((k1, k2, m))
        x = np.swapaxes(x, 0, 1)
        l2x = self.graph2.laplacian_mult(x.reshape((k2, -1))).reshape((k2, k1, m))
        l2x = np.swapaxes(l2x, 0, 1)
        return (l1x + l2x).reshape((-1, m))

    def laplacian_prox(
        self, v: npt.NDArray[np.float64], rho: float
    ) -> npt.NDArray[np.float64]:
        m = v.shape[-1]
        k1 = self.graph1.number_of_nodes()
        k2 = self.graph2.number_of_nodes()
        v = v.reshape((k1, k2, m))
        p1x = self.graph1.laplacian_prox(v.reshape((k1, -1)), rho * 2).reshape(
            (k1, k2, m)
        )
        v = np.swapaxes(v, 0, 1)
        p2x = self.graph2.laplacian_prox(v.reshape((k2, -1)), rho * 2).reshape(
            (k2, k1, m)
        )
        p2x = np.swapaxes(p2x, 0, 1)
        return (p1x + p2x).reshape((-1, m))

    @classmethod
    def multi_product(
        cls,
        graphs: list[RegularizationGraph[Node]],
    ) -> RegularizationGraph[Node] | CartesianProductOfGraphs[NestedNode, Node]:
        graph = graphs[0]
        for next_graph in graphs[1:]:
            graph = CartesianProductOfGraphs(graph, next_graph)
        return graph
