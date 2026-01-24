from abc import ABC, abstractmethod
from collections.abc import Hashable
from functools import cached_property

import attrs
import networkx as nx
import numpy as np

from stratified_models.linear_operator import Array
from stratified_models.simpler.model import Stratification
from stratified_models.simpler.scalar_function import ScalarFunction, TensorQuadForm


class RegularizationGraph(ABC):
    stratification: Stratification

    @abstractmethod
    def laplacian(self, axis: int, dims: tuple[int, ...]) -> ScalarFunction:
        pass

    def get_subnode_index(self, sub_node: Hashable) -> int:
        return self.stratification.get_subnode_index(sub_node)

    @property
    def size(self) -> int:
        return self.stratification.size

    @property
    def name(self) -> Hashable:
        return self.stratification.name


@attrs.frozen(kw_only=True)
class NetworkXRegularizationGraph(RegularizationGraph):
    stratification: Stratification
    graph: nx.Graph
    weight_key: str = "weight"

    @cached_property
    def laplacian_matrix(self) -> Array:
        mat = nx.laplacian_matrix(self.graph, weight=self.weight_key)
        # `networkx` returns a scipy sparse matrix/array; `TensorQuadForm` expects
        # a dense ndarray.
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        return np.asarray(mat, dtype=float)

    def laplacian(self, axis: int, dims: tuple[int, ...]) -> TensorQuadForm:
        return TensorQuadForm(
            a=self.laplacian_matrix,
            axis=axis,
            dims=dims,
        )
