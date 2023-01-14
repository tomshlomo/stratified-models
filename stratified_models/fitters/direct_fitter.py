from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from stratified_models.fitters.protocols import Node, NodeData, Theta
from stratified_models.regularization_graph.regularization_graph import (
    Name,
    RegularizationGraph,
)


class DirectFitter:
    def fit(
        self,
        nodes_data: dict[Node, NodeData],
        graph: RegularizationGraph[Node, Name],
        l2_reg: float,
        m: int,
    ) -> Theta:
        a, xy = self._build_lin_problem(
            nodes_data=nodes_data, graph=graph, l2_reg=l2_reg, m=m
        )
        theta = scipy.sparse.linalg.spsolve(a, xy)
        theta_df = pd.DataFrame(
            theta.reshape((-1, m)),
            index=graph.nodes(),
        )
        return theta_df

    def _build_lin_problem(
        self,
        nodes_data: dict[Node, NodeData],
        graph: RegularizationGraph[Node, Name],
        l2_reg: float,
        m: int,
    ) -> Tuple[scipy.sparse.csr_matrix, npt.NDArray[np.float64]]:
        k = graph.number_of_nodes()
        km = k * m
        a = scipy.sparse.eye(km, format="csr") * l2_reg
        xy = np.zeros(km)
        for node, node_data in nodes_data.items():
            i = graph.get_node_index(node)
            sl = slice(i * m, (i + 1) * m)
            a[sl, sl] += node_data.x.T @ node_data.x
            # diags.a
            xy[sl] = node_data.x.T @ node_data.y
        laplacian = graph.laplacian_matrix()
        laplacian = scipy.sparse.kron(laplacian, scipy.sparse.eye(m))
        a += laplacian
        return a, xy
