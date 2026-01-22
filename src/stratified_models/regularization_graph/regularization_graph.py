from abc import abstractmethod
from typing import Generic, Hashable, TypeVar

import pandas as pd

from stratified_models.scalar_function import Array, ScalarFunction

F = TypeVar("F", bound=ScalarFunction[Array], covariant=True)


class RegularizationGraph(Generic[F]):
    def __init__(self, nodes: pd.Index):
        self.nodes = nodes

    def get_node_index(self, node: Hashable) -> int:
        return self.nodes.get_loc(node)  # type:ignore[no-any-return]

    def number_of_nodes(self) -> int:
        return self.nodes.shape[0]  # type:ignore[no-any-return]

    def name(self) -> Hashable:
        return self.nodes.name  # type:ignore[no-any-return]

    @abstractmethod
    def laplacian(self, axis: int, dims: tuple[int, ...]) -> F:
        raise NotImplementedError
