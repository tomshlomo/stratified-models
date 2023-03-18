import networkx as nx
import pandas as pd
import scipy

from stratified_models.regularization_graph.regularization_graph import (
    RegularizationGraph,
)
from stratified_models.scalar_function import TensorQuadForm


class NetworkXRegularizationGraph(RegularizationGraph[TensorQuadForm]):
    # todo:rename to simply "WEIGHT_KEY", and change value to simply "weight"
    # todo: should be a parameter to constructor, with default value "weight"
    LAPLACE_REG_PARAM_KEY = "laplace_reg_param"

    def __init__(self, graph: nx.Graph, name: str):
        super().__init__(nodes=pd.Index(graph.nodes, name=name))
        self.graph = graph
        self._laplacian_matrix = nx.laplacian_matrix(
            self.graph, weight=self.LAPLACE_REG_PARAM_KEY
        )

    def laplacian(self, axis: int, dims: tuple[int, ...]) -> TensorQuadForm:
        return TensorQuadForm(
            a=self.laplacian_matrix().toarray(),
            axis=axis,
            dims=dims,
        )

    def laplacian_matrix(self) -> scipy.sparse.spmatrix:
        return self._laplacian_matrix
