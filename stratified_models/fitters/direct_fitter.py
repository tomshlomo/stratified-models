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


class DirectFitter:
    def fit(
        self,
        nodes_data: dict[Node, NodeData],
        graphs: dict[str, nx.Graph],
        l2_reg: float,
        m: int,
    ) -> Theta:
        graph = cartesian_product(graphs.values())
        a, xy = self._build_lin_problem(
            nodes_data=nodes_data, graph=graph, l2_reg=l2_reg, m=m
        )
        theta = scipy.sparse.linalg.spsolve(a, xy)
        theta_df = pd.DataFrame(
            theta.reshape((-1, m)),
            index=pd.MultiIndex.from_tuples(graph.nodes, names=graphs.keys()),
        )
        return theta_df

    def _build_lin_problem(
        self,
        nodes_data: dict[Node, NodeData],
        graph: nx.Graph,
        l2_reg: float,
        m: int,
    ) -> (scipy.sparse.csr_matrix, npt.NDArray,):
        k = graph.number_of_nodes()
        km = k * m
        a = scipy.sparse.eye(km, format="csr") * l2_reg
        xy = np.zeros(km)
        node2index = {node: i for i, node in enumerate(graph.nodes)}
        for node, node_data in nodes_data.items():
            i = node2index[node]
            sl = slice(i * m, (i + 1) * m)
            a[sl, sl] += node_data.x.T @ node_data.x
            xy[sl] = node_data.x.T @ node_data.y
        laplacian = nx.laplacian_matrix(graph, weight=LAPLACE_REG_PARAM_KEY)
        laplacian = scipy.sparse.kron(laplacian, scipy.sparse.eye(m))
        a += laplacian
        return a, xy
