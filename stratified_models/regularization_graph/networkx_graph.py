from typing import Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from numpy import typing as npt

from stratified_models.regularization_graph.regularization_graph import (
    Node,
    RegularizationGraph,
)


class NetworkXRegularizationGraph(RegularizationGraph[Node]):
    # todo:rename to simply "WEIGHT_KEY", and change value to simply "weight"
    LAPLACE_REG_PARAM_KEY = "laplace_reg_param"

    def __init__(self, graph: nx.Graph, name: str):
        super().__init__(nodes=pd.Index(graph.nodes, name=name))
        self.graph = graph
        self._laplacian_eig_cache: Optional[
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = None
        self._laplacian_matrix = nx.laplacian_matrix(
            self.graph, weight=self.LAPLACE_REG_PARAM_KEY
        )

    def laplacian_eig(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self._laplacian_eig_cache is None:
            # todo: use scipy.sparse.eigh?
            self._laplacian_eig_cache = np.linalg.eigh(  # type:ignore[assignment]
                self.laplacian_matrix().toarray()
            )
        return self._laplacian_eig_cache  # type:ignore[return-value]

    def laplacian_matrix(self) -> scipy.sparse.spmatrix:
        return self._laplacian_matrix

    def laplacian_mult(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._laplacian_matrix @ x

    def laplacian_prox(
        self, v: npt.NDArray[np.float64], rho: float
    ) -> npt.NDArray[np.float64]:
        """
        argmin theta tr(theta' L theta) + rho/2 ||theta - v||^2
        2 L theta + rho ( theta - v) = 0
        theta = (2 L + rho I)^-1 rho v
        = (rho(2 L / rho + I))^-1 rho v
        = (L 2/rho + I)^-1 v
        :param v:
        :param rho:
        :return:
        """
        w, u = self.laplacian_eig()
        d = 1 / (1 + w * (2 / rho))
        # todo: cache optimal einsum path
        return np.einsum(
            "mi,i,pi,pj->mj", u, d, u, v, optimize="optimal"
        )  # type:ignore[no-any-return]
