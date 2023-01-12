from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from stratified_models.fitters.protocols import (
    LAPLACE_REG_PARAM_KEY,
    Node,
    NodeData,
    Theta,
)
from stratified_models.utils.networkx_utils import cartesian_product


@dataclass
class CGFitter:
    tol: float = 1e-6
    max_iter: Optional[int] = None

    def fit(
        self,
        nodes_data: dict[Node, NodeData],
        graphs: dict[str, nx.Graph],
        l2_reg: float,
        m: int,
    ) -> Theta:
        graph = cartesian_product(graphs.values())
        indexed_nodes_data = [
            (i, nodes_data[node])
            for i, node in enumerate(graph.nodes)
            if node in nodes_data
        ]

        a = LinearOperator.from_data(
            nodes_data=indexed_nodes_data,
            graph=graph,
            l2_reg=l2_reg,
            m=m,
        )
        xy = self._build_xy(
            nodes_data=indexed_nodes_data,
            graph=graph,
            m=m,
        )
        theta, info = scipy.sparse.linalg.cg(
            a, xy.ravel(), tol=self.tol, maxiter=self.max_iter
        )
        theta_df = pd.DataFrame(  # todo: should be a common function
            theta.reshape((-1, m)),
            index=pd.MultiIndex.from_tuples(graph.nodes, names=graphs.keys()),
        )
        return theta_df

    def _build_xy(
        self,
        nodes_data: list[Tuple[int, NodeData]],
        graph: nx.Graph,
        m: int,
    ) -> npt.NDArray[np.float64]:
        k = graph.number_of_nodes()
        xy = np.zeros((k, m))
        for i, node_data in nodes_data:
            xy[i] = node_data.x.T @ node_data.y
        return xy


class LinearOperator(scipy.sparse.linalg.LinearOperator):  # type:ignore[misc]
    def __init__(
        self, q: npt.NDArray[np.float64], laplacian: scipy.sparse.csr_matrix
    ) -> None:
        self.k, self.m, _ = q.shape
        self.q = q
        self.laplacian = laplacian
        mk = self.m * self.k
        self.shape = (mk, mk)
        self.dtype = q.dtype

    @classmethod
    def from_data(
        cls,
        nodes_data: list[Tuple[int, NodeData]],
        graph: nx.Graph,
        l2_reg: float,
        m: int,
    ) -> LinearOperator:
        k = graph.number_of_nodes()
        q = np.tile(np.eye(m) * l2_reg, (k, 1, 1))
        xy = np.zeros((k, m))
        for i, node_data in nodes_data:
            q[i] += node_data.x.T @ node_data.x
            xy[i] = node_data.x.T @ node_data.y
        laplacian = nx.laplacian_matrix(graph, weight=LAPLACE_REG_PARAM_KEY)
        return LinearOperator(q=q, laplacian=laplacian)

    def _matvec(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x = x.reshape((self.k, self.m))
        qx = np.einsum("kim,ki->km", self.q, x)
        lx = self.laplacian @ x
        return (qx + lx).ravel()  # type:ignore[no-any-return]
