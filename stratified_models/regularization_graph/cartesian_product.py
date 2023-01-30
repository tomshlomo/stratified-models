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

    def laplacian_mult(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        (A*B is kron(A, B))
        L x = (I * L2 + L1 * I) x
        = (I * L2) x + (L1 * I) x
        = vec( L2 mat(x) ) + vec( mat(x) L1 )
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
        """
        u = u1 x u2
        w = w1 x w2
        theta = (tL + I)^-1 v
        u d u' v
        d = (tw + I)^-1 = d1 x d2
        di = (twi + I)^-1
        prox = (u1 x u2) (d1 x d2) (u1 x u2)' v
        = (u1 d1 u1') x (u2 d2 u2') v
        = (prox1 x prox2) v
        = vec(prox2 V prox1')
        """
        m = v.shape[-1]
        k1 = self.graph1.number_of_nodes()
        k2 = self.graph2.number_of_nodes()
        k = self.number_of_nodes()
        v = v.reshape((k1, k2, m))
        # v = v.reshape((k2, -1))  # k1, k2*m
        v = np.swapaxes(v, 0, 1)
        v = v.reshape((k2, -1))
        v = self.graph2.laplacian_prox(v, rho)  # k1, k2*m
        # todo: can be replaced by a single eingsum, probably
        v = np.swapaxes(v.reshape((k2, k1, -1)), 0, 1).reshape((k1, -1))
        v = self.graph1.laplacian_prox(v, rho)
        return v.reshape((k, m))

    @classmethod
    def multi_product(
        cls,
        graphs: list[RegularizationGraph[Node]],
    ) -> RegularizationGraph[Node] | CartesianProductOfGraphs[NestedNode, Node]:
        graph = graphs[0]
        for next_graph in graphs[1:]:
            graph = CartesianProductOfGraphs(graph, next_graph)
        return graph
