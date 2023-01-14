from typing import Tuple, TypeVar

import numpy as np
import pandas as pd
import scipy
from numpy import typing as npt

from stratified_models.regularization_graph.regularization_graph import (
    RegularizationGraph,
    RegularizationGraphWithCachedNodes,
)

Node1 = TypeVar("Node1")
Node2 = TypeVar("Node2")
Name1 = TypeVar("Name1")
Name2 = TypeVar("Name2")


class CartesianProductOfGraphs(
    RegularizationGraphWithCachedNodes[Tuple[Node1, Node2], Tuple[Name1, Name2]]
):
    def __init__(
        self,
        graph1: RegularizationGraph[Node1, Name1],
        graph2: RegularizationGraph[Node2, Name2],
    ) -> None:
        super().__init__(
            nodes=pd.MultiIndex.from_product(
                [graph1.nodes(), graph2.nodes()],
                names=[graph1.name, graph2.name],
            )
        )
        self.graph1 = graph1
        self.graph2 = graph2

    def laplacian_matrix(self) -> scipy.sparse.csr_matrix:
        lap1 = self.graph1.laplacian_matrix()
        lap2 = self.graph2.laplacian_matrix()
        return scipy.sparse.kronsum(lap2, lap1)

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
