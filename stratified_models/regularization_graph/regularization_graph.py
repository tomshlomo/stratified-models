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
        pass

    # @abstractmethod
    # def laplacian_operator(self, axis: int, dims: tuple[int, ...]) -> LinearOperator:
    #     pass
    #
    # @abstractmethod
    # def laplacian_matrix(self) -> scipy.sparse.spmatrix:
    #     pass

    # def laplacian_prox_matrix(self, rho: float) -> npt.NDArray[np.float64]:
    #     return self.laplacian_prox(np.eye(self.number_of_nodes()), rho)

    # @abstractmethod
    # def laplacian_mult(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    #     pass
    #
    # def laplacian_quad_form(self, x: npt.NDArray[np.float64]) -> float:
    #     return x.ravel() @ self.laplacian_mult(x).ravel()
    #
    # def laplacian_prox(
    #     self, v: npt.NDArray[np.float64], rho: float
    # ) -> npt.NDArray[np.float64]:
    #     v = self.gft(v)
    #     v /= self.spectrum()[:, np.newaxis] * (2 / rho) + 1.0
    #     return self.igft(v)
    #
    # @abstractmethod
    # def gft(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    #     pass
    #
    # @abstractmethod
    # def igft(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    #     pass
    #
    # @abstractmethod
    # def spectrum(self) -> npt.NDArray[np.float64]:
    #     pass
