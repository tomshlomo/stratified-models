from abc import ABC, abstractmethod
from collections.abc import Hashable
from functools import cached_property

import attrs
import networkx as nx
from scipy.sparse import sparray

from stratified_models.simpler.model import Stratification
from stratified_models.simpler.scalar_function import (
    ScalarFunction,
    SparseQuadraticForm,
)


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
    def laplacian_matrix(self) -> sparray:
        return nx.laplacian_matrix(self.graph, weight=self.weight_key)

    def laplacian(self, axis: int, dims: tuple[int, ...]) -> SparseQuadraticForm:
        return SparseQuadraticForm(
            a=self.laplacian_matrix,
            axis=axis,
            dims=dims,
        )
