from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy

from stratified_models.fitters.protocols import Node, NodeData, Theta
from stratified_models.regularization_graph.regularization_graph import (
    RegularizationGraph,
)

"""
cost(theta) = sum_k f_k(theta_k) + laplacian(theta)
where:
f_k(theta_k) = |x_k @ theta_k - y_k|^2 + l2_reg * |theta_k|^2 =
             = theta_k ' (x_k' x_k + l2_reg * I) theta_k - 2 x_k' y_k + |y_k|^2
             = theta_k' q_k theta_k - 2 c_k' theta_k + d_k
q_k = (x_k' x_k + l2_reg * I)
c_k = x_k' y_k
d_k = |y_k|^2


laplacian(theta) = tr(theta' L theta) / 2

ADMM:
first, rewrite as:
minimize sum_k f_k(theta_k) + laplacian(theta_tilde) + rho/2 |theta - theta_tilde|^2
s.t theta_tilde_k = theta_k for all k

derive the lagrangian w.r.t dual variable y:
L(theta, theta_tilde, y) =
sum_k f_k(theta_k)
+ laplacian(theta_tilde)
+ rho/2 |theta - theta_tilde|^2
+ tr(y' (theta - theta_tilde))

let:
rho = 1/t
u = y * t = y / rho

the dual function is:
g(y) = min_theta, theta_tilde L(theta, theta_tilde, y)

ADMM:
(1) theta <- argmin_theta (L(theta, theta_tilde, y)
(2) theta_tilde <- argmin_theta_tilde (L(theta, theta_tilde, y)
(3) y += grad_y L(theta, theta_tilde, y) * rho

(1)_k can be written as:
argmin_theta_k f_k(theta_k) + rho/2 |theta_k - theta_tilde_k + y_k*t|^2
= prox_t f_k (theta_tilde_k - y_k * t)
= prox_t f_k (theta_tilde_k - u_k)

for the case that
f_k(theta_k)/2 = theta_k' q_k theta_k/2  + c_k + d_k/2,
we have that:
prox_t f_k(v) = argmin_theta_k theta_k' q_k theta_k  - 2 c_k + d_k + rho/2 |theta_k-v|^2
taking the gradient=0 gives:
2 q_k theta_k - 2 c_k + rho(theta_k - v) = 0
(2 q_k + rho * I)theta_k = rho v + 2 c_k
theta_k = (q_k + rho/2 * I)^-1  (rho/2 * v + c_k)

(2) can be written as:
argmin_theta_tilde laplacian(theta_tilde) + rho/2 |theta_tilde - theta - y_k * t|^2
=prox_tl(theta + u)

since laplacian(theta_tilde) = tr(theta' L theta)/2, we have that:
prox_tl (v) = argmin_theta_tilde (
tr(theta_tilde' L theta_tilde)/2
+ rho/2 |theta_tilde - v|^2
)
taking the gradient=0 gives:
L theta_tilde + rho (theta_tilde - v) = 0
(L + rho * I) theta_tilde = rho * v
theta_tilde = (L + rho * I)^-1 rho * v
            = (tL + I)^-1 v


(3) can be written as:
grad_y L(theta, theta_tilde, y) * rho =
(theta - theta_tilde) * rho
y <- y + rho * (theta - theta_tilde)
u <- u + theta - theta_tilde
"""


@dataclass
class ADMMFitter:
    tol: float = 1e-6
    max_iter: int = 1000

    def fit(
        self,
        nodes_data: dict[Node, NodeData],
        graph: RegularizationGraph,
        # graphs: dict[str, nx.Graph],
        l2_reg: float,
        m: int,
    ) -> Theta:
        # graph = cartesian_product(graphs.values())
        k = graph.number_of_nodes()
        instance = QuadraticStratifiedProblem.from_data(
            nodes_data=nodes_data,
            graph=graph,
            l2_reg=l2_reg,
            m=m,
        )
        u = np.zeros((k, m))
        theta = np.zeros((k, m))
        theta_tilde = np.zeros((k, m))
        rho = 1.0
        cost = np.zeros((self.max_iter + 1, 2))
        cost[0, 0] = instance.eval(theta)
        cost[0, 1] = instance.eval(theta_tilde)
        for i in range(1, self.max_iter + 1):
            theta = instance.prox_f(theta_tilde - u, rho)
            theta_tilde = instance.prox_l(theta + u, rho)
            u += theta - theta_tilde
            cost[i, 0] = instance.eval(theta)
            cost[i, 1] = instance.eval(theta_tilde)

        theta_df = pd.DataFrame(  # todo: should be a common function
            theta,
            index=graph.nodes(),
        )
        return theta_df


class QuadraticStratifiedProblem:  # todo:rename.
    def __init__(
        self,
        q: npt.NDArray[np.float64],  # k, m, m
        c: npt.NDArray[np.float64],  # k, m
        d: float,
        laplacian: scipy.sparse.csr_matrix,  # k, k
    ) -> None:
        self.q = q
        self.c = c
        self.d = d
        self.laplacian = laplacian
        self.k, self.m, _ = q.shape
        self._q_eig_cache: Optional[
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = None
        self._l_eig_cache: Optional[
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = None

    def eval(self, theta: npt.NDArray[np.float64]) -> float:
        """
        sum_k theta_k' q_k theta_k - 2 c_k' theta_k + d_k
        +
        tr(theta' L theta) / 2
        """
        q_theta = np.einsum("kim,ki->km", self.q, theta)
        q_theta_plus_lx = q_theta + self.laplacian @ theta / 2
        return (  # type:ignore[no-any-return]
            theta.ravel() @ (-2 * self.c + q_theta_plus_lx).ravel() + self.d
        )

    @classmethod
    def from_data(
        cls,
        nodes_data: dict[Node, NodeData],
        graph: RegularizationGraph[Node],
        l2_reg: float,
        m: int,
    ) -> QuadraticStratifiedProblem:
        k = graph.number_of_nodes()
        q = np.tile(np.eye(m) * l2_reg, (k, 1, 1))
        xy = np.zeros((k, m))
        d = 0.0
        for node, node_data in nodes_data.items():
            i = graph.get_node_index(node)
            q[i] += node_data.x.T @ node_data.x
            xy[i] = node_data.x.T @ node_data.y
            d += float(node_data.y @ node_data.y)
        laplacian = graph.laplacian_matrix()
        return QuadraticStratifiedProblem(q=q, c=xy, d=d, laplacian=laplacian)

    def prox_f(self, v: npt.NDArray[np.float64], rho: float) -> npt.NDArray[np.float64]:
        """
        let let uk w_k u_k' = q_k, then:
        theta_k = (q_k + rho/2 * I)^-1  (rho/2 * v + c_k)
        = u_k (w_k + rho/2 I) u_k' (rho/2 * v + c_k)
        """
        hrho = rho / 2
        b = hrho * v + self.c
        if self._q_eig_cache is None:
            self._q_eig_cache = np.linalg.eigh(self.q)
        w, u = self._q_eig_cache
        d = 1 / (w + hrho)
        # todo: cache optimal einstein path
        return np.einsum("kmi,ki,kpi,kp->km", u, d, u, b)  # type:ignore[no-any-return]

    def prox_l(self, v: npt.NDArray[np.float64], rho: float) -> npt.NDArray[np.float64]:
        """
        (laplacian / rho + I)^-1 v =
        u (1 + w/rho)^-1 u' v
        :param v:
        :param rho:
        :return:
        """
        if self._l_eig_cache is None:
            self._l_eig_cache = np.linalg.eigh(
                self.laplacian.toarray()
            )  # type:ignore[assignment]
        w, u = self._l_eig_cache  # type:ignore[misc]
        d = 1 / (1 + w / rho)
        # todo: cache optimal einstein path
        return np.einsum("mi,i,pi,pj->mj", u, d, u, v)  # type:ignore[no-any-return]
