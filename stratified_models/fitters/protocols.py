from __future__ import annotations

# todo: rename this file
from dataclasses import dataclass
from functools import partial
from typing import Callable, Generic, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from stratified_models.regularization_graph.regularization_graph import (
    Node,
    RegularizationGraph,
)


@dataclass
class Theta(Generic[Node]):
    df: pd.DataFrame


@dataclass
class NodeData:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]


@dataclass
class QuadraticStratifiedLinearRegressionProblem(Generic[Node]):
    """
    sum_k ||X_k theta_k - y_k||^2 + gamma ||Theta||^2 + tr(Theta' L Theta)
    =
    theta' A theta - 2 c' theta + d
    where:
    theta = Theta.ravel()
    A = (Q + gamma I + L)
    Q is block diagonal, block k = x_k' x_k
    block k of c = x_k' y_k
    d = sum_k ||y_k||^2

    The argmin is given by:
    theta_opt = A^{-1} c

    The min is given by:
    c' A^-1 A A^-1 c - 2 c' A^-1 c + d
    = c' A^-1 c - 2 c' A^-1 c + d
    = d - c' A^-1 c
    = d - c' theta_opt
    """

    nodes_data: dict[Node, NodeData]
    graph: RegularizationGraph[Node]
    l2_reg: float
    m: int

    def build_a_c_d(
        self,
    ) -> tuple[LinearOperator, npt.NDArray[np.float64], float]:
        k = self.graph.number_of_nodes()
        xy = np.zeros((k, self.m))
        q = np.zeros((k, self.m, self.m))
        # q = np.tile(np.eye(self.m) * self.l2_reg, (k, 1, 1))
        # laplacian = self.graph.laplacian_matrix()
        d = 0.0
        for node, node_data in self.nodes_data.items():
            i = self.graph.get_node_index(node)
            q[i] += node_data.x.T @ node_data.x
            xy[i] = node_data.x.T @ node_data.y
            d += node_data.y @ node_data.y
        return (
            LinearOperator(
                q=BlockDiagonalLinearOperator(q=q),
                l2_reg=self.l2_reg,
                graph=self.graph,
                # q=q,
                # half_laplacian=laplacian / 2,
            ),
            xy,
            d,
        )


class LinearOperator:  # type:ignore[misc]
    # todo: input should be graph, and there should be a laplacian_mult
    #  method for RegularizationGraph
    # todo: should get problem data and regularization object, not q
    def __init__(
        self,
        q: BlockDiagonalLinearOperator,
        l2_reg: float,
        graph: RegularizationGraph[Node],
    ) -> None:
        # self.k, self.m, _ = q.shape
        self.q = q
        self.l2_reg = l2_reg
        self.graph = graph
        self.k = self.q.k
        self.m = self.q.m
        self.dtype = self.q.dtype

        # self.half_laplacian = half_laplacian
        # mk = self.m * self.k

    def matvec(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = self.q.matvec(x)
        out += self.graph.laplacian_mult(x)
        out += x * self.l2_reg
        return out

    def ravel_matvec(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x = x.reshape((self.k, self.m))
        return self.matvec(x).ravel()

    def q_plus_reg(self) -> BlockDiagonalLinearOperator:
        return BlockDiagonalLinearOperator(
            q=self.q.q + self.l2_reg * np.eye(self.m),
        )

    def as_matrix(self) -> scipy.sparse.spmatrix:
        a = self.q.as_matrix()
        a += scipy.sparse.kron(self.graph.laplacian_matrix(), scipy.sparse.eye(self.m))
        a += scipy.sparse.eye(self.k * self.m) * self.l2_reg
        return a

    def as_scipy_linear_operator(self) -> CallbackBasedScipyLinearOperator:
        return CallbackBasedScipyLinearOperator(
            matvec=self.ravel_matvec,
            shape=(self.k * self.m, self.k * self.m),
            dtype=self.dtype,
        )


class BlockDiagonalLinearOperator:
    def __init__(
        self,
        q: npt.NDArray[np.float64],
    ) -> None:
        self.k, self.m, _ = q.shape
        self.q = q
        self.dtype = self.q.dtype
        # mk = self.m * self.k
        # self.shape = (mk, mk)
        # self.dtype = q.dtype

    def matvec(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # x = x.reshape((self.k, self.m))
        return np.einsum("kim,ki->km", self.q, x)
        # return qx.ravel()  # type:ignore[no-any-return]

    def ravel_matvec(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x = x.reshape((self.k, self.m))
        return self.matvec(x).ravel()

    def as_matrix(self) -> scipy.sparse.spmatrix:
        return scipy.sparse.block_diag(self.q)

    def as_scipy_linear_operator(self) -> CallbackBasedScipyLinearOperator:
        return CallbackBasedScipyLinearOperator(
            matvec=partial(self.ravel_matvec, self),
            shape=(self.k * self.m, self.k * self.m),
            dtype=self.dtype,
        )

    def quad_form_prox(self, v, rho):
        """
        argmin x' a x + rho/2 |x-v|^2
        2 a x + rho (x-v) = 0
        (2a + rho I) x = rho v
        x = (2/rho a + I)^-1 v
        :param v:
        :param rho:
        :return:
        """
        p = 2 / rho * self.q + np.eye(self.m)
        x = np.zeros((self.k, self.m))
        for i in range(self.k):
            x[i] = np.linalg.solve(p[i], v[i])
        return x

    def traces(self):
        # todo: rewrite using einsum
        x = np.zeros(self.k)
        for i in range(self.k):
            x[i] = np.trace(self.q[i])
        return x


class CallbackBasedScipyLinearOperator(scipy.sparse.linalg.LinearOperator):
    def __init__(
        self,
        matvec: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        shape: tuple[int, int],
        dtype: type,
    ):
        self._matvec = matvec
        self.shape = shape
        self.dtype = dtype


@dataclass
class Costs:
    loss: float
    local_regularization: float
    laplacian_regularization: float

    @classmethod
    def from_problem_and_theta(
        cls,
        problem: QuadraticStratifiedLinearRegressionProblem[Node],
        theta: Theta[Node],
    ) -> Costs:
        loss = 0.0
        theta_df = theta.df.set_index(pd.MultiIndex.from_tuples(theta.df.index))
        for node, node_data in problem.nodes_data.items():
            y_pred = node_data.x @ theta_df.loc[node, :].values
            d = y_pred - node_data.y
            loss += d @ d
        local_reg = theta.df.values.ravel() @ theta.df.values.ravel() * problem.l2_reg
        laplace_reg = problem.graph.laplacian_quad_form(theta_df.values)
        return Costs(
            loss=loss,
            local_regularization=local_reg,
            laplacian_regularization=laplace_reg,
        )

    def total(self) -> float:
        return self.loss + self.local_regularization + self.laplacian_regularization


class StratifiedLinearRegressionFitter(Protocol[Node]):  # pragma: no cover
    def fit(
        self,
        problem: QuadraticStratifiedLinearRegressionProblem,
    ) -> tuple[Theta[Node], float]:
        pass
