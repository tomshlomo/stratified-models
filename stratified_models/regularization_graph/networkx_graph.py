from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from numpy import typing as npt

from stratified_models.fitters.protocols import Node
from stratified_models.regularization_graph.regularization_graph import (
    RegularizationGraphWithCachedNodes,
)


class NetworkXRegularizationGraph(RegularizationGraphWithCachedNodes[Node, str]):
    # todo:rename to simply "WEIGHT_KEY", and change value to simply "weight"
    LAPLACE_REG_PARAM_KEY = "laplace_reg_param"

    def __init__(self, graph: nx.Graph, name: str):
        super().__init__(nodes=pd.Index(graph.nodes, name=name))
        self.graph = graph
        self._laplacian_eig_cache = None

    def laplacian_eig(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self._laplacian_eig_cache is None:
            # todo: use scipy.sparse.eigh?
            self._laplacian_eig_cache = np.linalg.eigh(
                self.laplacian_matrix().toarray()
            )
        return self._laplacian_eig_cache

    def laplacian_matrix(self) -> scipy.sparse.csr_matrix:
        return nx.laplacian_matrix(self.graph, weight=self.LAPLACE_REG_PARAM_KEY)

    def laplacian_prox(
        self, v: npt.NDArray[np.float64], rho: float
    ) -> npt.NDArray[np.float64]:
        w, u = self.laplacian_eig()
        d = 1 / (1 + w / rho)
        # todo: cache optimal einsum path
        return np.einsum("mi,i,pi,pj->mj", u, d, u, v)  # type:ignore[no-any-return]
