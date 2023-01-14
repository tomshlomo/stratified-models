from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
from pandas.core.indexes.frozen import FrozenList

Node = TypeVar("Node")
Name = TypeVar("Name")


class RegularizationGraph(Generic[Node, Name]):
    def __init__(self, name: Name):
        self.name = name

    @abstractmethod
    def nodes(self) -> pd.Index:
        pass

    @abstractmethod
    def get_node_index(self, node: Node) -> int:
        pass

    @abstractmethod
    def number_of_nodes(self) -> int:
        pass

    @abstractmethod
    def laplacian_matrix(self) -> scipy.sparse.csr_matrix:
        pass

    @abstractmethod
    def laplacian_prox(
        self, v: npt.NDArray[np.float64], rho: float
    ) -> npt.NDArray[np.float64]:
        pass


# todo: remove the above and leave only this implementation
class RegularizationGraphWithCachedNodes(
    RegularizationGraph[Node, FrozenList[Name]],
    ABC,
):
    def __init__(self, nodes: pd.Index):
        super().__init__(name=nodes.names)
        self._nodes = nodes

    def get_node_index(self, node: Node) -> int:
        return self._nodes.get_loc(node)

    def nodes(self) -> pd.Index:
        return self._nodes

    def number_of_nodes(self) -> int:
        return self._nodes.shape[0]
