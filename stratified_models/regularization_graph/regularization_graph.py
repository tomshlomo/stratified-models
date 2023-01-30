from abc import abstractmethod
from typing import Generic, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

Node = TypeVar("Node")
Name = Union[str, Tuple["Name", "Name"]]


class RegularizationGraph(Generic[Node]):
    def __init__(self, nodes: pd.Index):
        self.nodes = nodes

    def get_node_index(self, node: Node) -> int:
        return self.nodes.get_loc(node)  # type:ignore[no-any-return]

    def number_of_nodes(self) -> int:
        return self.nodes.shape[0]  # type:ignore[no-any-return]

    def name(self) -> Name:
        return self.nodes.name  # type:ignore[no-any-return]

    @abstractmethod
    def laplacian_matrix(self) -> scipy.sparse.spmatrix:
        pass

    @abstractmethod
    def laplacian_mult(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

    def laplacian_quad_form(self, x: npt.NDArray[np.float64]) -> float:
        return x.ravel() @ self.laplacian_mult(x).ravel()

    @abstractmethod
    def laplacian_prox(
        self, v: npt.NDArray[np.float64], rho: float
    ) -> npt.NDArray[np.float64]:
        pass
