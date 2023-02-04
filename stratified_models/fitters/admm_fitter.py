from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from stratified_models.fitters.protocols import (
    Costs,
    LinearOperator,
    QuadraticStratifiedLinearRegressionProblem,
    Theta,
)
from stratified_models.regularization_graph.regularization_graph import (
    Node,
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


laplacian(theta) = tr(theta' L theta)

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
(2/ rho q_k + I) theta_k = v + 2/rho c_k
theta_k = (q_k + rho/2 * I)^-1  (rho/2 * v + c_k)

(2) can be written as:
argmin_theta_tilde laplacian(theta_tilde) + rho/2 |theta_tilde - theta - y_k * t|^2
=prox_tl(theta + u)

since laplacian(theta_tilde) = tr(theta' L theta), we have that:
prox_tl (v) = argmin_theta_tilde (
tr(theta_tilde' L theta_tilde)
+ rho/2 |theta_tilde - v|^2
)
taking the gradient=0 gives:
2 L theta_tilde + rho (theta_tilde - v) = 0
(2 L + rho * I) theta_tilde = rho * v
theta_tilde = (2L + rho * I)^-1 rho * v
            = (2tL + I)^-1 v


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
        problem: QuadraticStratifiedLinearRegressionProblem[Node],
    ) -> tuple[Theta[Node], Costs]:
        start = time()
        k = problem.graph.number_of_nodes()
        m = problem.m
        a, c, d = problem.build_a_c_d()
        f = a.q_plus_reg()
        u = np.zeros((k, m))

        def calc_cost(theta):
            return Costs.from_problem_and_theta(
                problem,
                Theta(
                    pd.DataFrame(
                        theta,
                        index=problem.graph.nodes,
                    )
                ),
            ).total()

        theta = np.zeros((k, m))
        theta_tilde = np.zeros((k, m))

        rho = self.suggest_rho(a)
        cost = np.zeros((self.max_iter + 1, 2))
        cost[0, :] = calc_cost(theta)
        best_cost = np.min(cost[0, 0])
        best_theta = theta
        print(
            f"{'i':<3} | {'is_new_best':<15} {'best_cost':<15} {'d_cost0':<15} "
            f"{'d_cost1':<15} {'rho':<15} {'dual_norm':<15} {'primal_norm':<15} "
            f"time"
        )
        for i in range(1, self.max_iter + 1):
            theta = f.quad_form_prox(theta_tilde - u + (2 / rho) * c, rho)
            dual_residual = theta - theta_tilde
            theta_tilde = problem.graph.laplacian_prox(theta + u, rho)
            primal_residual = theta - theta_tilde
            u += primal_residual

            cost[i, 0] = calc_cost(theta)
            cost[i, 1] = calc_cost(theta_tilde)
            is_new_best = False
            if cost[i, 0] < best_cost and cost[i, 0] < cost[i, 1]:
                best_cost = cost[i, 0]
                best_theta = theta
                is_new_best = True
            elif cost[i, 1] < best_cost:
                best_cost = cost[i, 1]
                best_theta = theta_tilde
                is_new_best = True

            dual_residual_norm = np.linalg.norm(dual_residual.ravel()) * rho
            primal_residual_norm = np.linalg.norm(primal_residual.ravel())

            print(
                f"{i:<3d} | {is_new_best:<15} {best_cost:<15.3e} "
                f"{cost[i, 0] - best_cost:<15.3e} "
                f"{cost[i, 1] - best_cost:<15.3e} "
                f"{rho:<15.3e} {dual_residual_norm:<15.3e} "
                f"{primal_residual_norm:<15.3e} {time() - start:4.3f}"
            )

            # if primal_residual_norm > 5 * dual_residual_norm:
            #     rho *= 2
            #     u /= 2
            # elif dual_residual_norm > 5 * primal_residual_norm:
            #     rho /= 2
            #     u *= 2

        theta_df = pd.DataFrame(  # todo: should be a common function
            best_theta,
            index=problem.graph.nodes,
        )
        return Theta(df=theta_df), best_cost

    def suggest_rho(self, a: LinearOperator) -> float:
        delta = np.sum(a.graph.laplacian_matrix().diagonal()) / (a.k * a.m)
        lambda_min = a.l2_reg
        lambda_max = np.max(a.q.traces()) + a.l2_reg
        if delta < lambda_min:
            rho = np.sqrt(lambda_min * delta)
        elif delta > lambda_max:
            rho = np.sqrt(lambda_max * delta)
        else:
            rho = delta
        return rho


class QuadraticStratifiedProblem:  # todo:delete.
    def __init__(
        self,
        q: npt.NDArray[np.float64],  # k, m, m
        c: npt.NDArray[np.float64],  # k, m
        d: float,
        graph: RegularizationGraph[Node],  # k, k
    ) -> None:
        self.q = q
        self.c = c
        self.d = d
        self.graph = graph
        self.laplacian = graph.laplacian_matrix()
        self.k, self.m, _ = q.shape
        self._q_eig_cache: Tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64]
        ] = np.linalg.eigh(self.q)

    def eval(self, theta: npt.NDArray[np.float64]) -> float:
        """
        sum_k theta_k' q_k theta_k - 2 c_k' theta_k + d_k
        +
        tr(theta' L theta) / 2
        """
        # z = self.d
        # for kk in range(self.k):
        #     z += theta[kk].T @ self.q[kk] @ theta[kk].T - 2 * self.c[kk].T @ theta[kk]
        # z += np.sum(theta * (self.laplacian @ theta)) / 2

        q_theta = np.einsum("kim,ki->km", self.q, theta)
        q_theta_plus_lx = q_theta + self.laplacian @ theta
        # todo: implement a laplacian_mult method on RegularizaionGraph,
        #  and remove laplacian from object
        return (  # type:ignore[no-any-return]
            theta.ravel() @ (-2 * self.c + q_theta_plus_lx).ravel() + self.d
        )

    @classmethod
    def from_data(
        cls,
        problem: QuadraticStratifiedLinearRegressionProblem[Node],
    ) -> QuadraticStratifiedProblem:
        k = problem.graph.number_of_nodes()
        m = problem.m
        q = np.tile(np.eye(m) * problem.l2_reg, (k, 1, 1))
        xy = np.zeros((k, m))
        d = 0.0
        for node, node_data in problem.nodes_data.items():
            i = problem.graph.get_node_index(node)
            q[i] += node_data.x.T @ node_data.x
            xy[i] = node_data.x.T @ node_data.y
            d += float(node_data.y @ node_data.y)
        return QuadraticStratifiedProblem(q=q, c=xy, d=d, graph=problem.graph)

    def prox_f(self, v: npt.NDArray[np.float64], rho: float) -> npt.NDArray[np.float64]:
        """
        min x'qx -2c'x + rho/2|x-v|^2
        2qx - 2c + rho (x-v) = 0
        qx - c + rho/2 (x-v) = 0
        (q + rho/2 I) x = c + rho/2 v
        x = (q + rho/2 I)^-1 (c + rho/2 v)

        let let uk w_k u_k' = q_k, then:
        theta_k = (q_k + rho/2 * I)^-1  (rho/2 * v + c_k)
        = u_k (w_k + rho/2 I)^-1 u_k' (rho/2 * v + c_k)
        """
        hrho = rho / 2
        b = hrho * v + self.c
        w, u = self._q_eig_cache
        d = 1 / (w + hrho)
        # todo: cache optimal einstein path
        # z = np.zeros((self.k, self.m))
        # for kk in range(self.k):
        #     z[kk] = u[kk] @ np.diag(d[kk]) @ u[kk].T @ b[kk]
        return np.einsum(
            "kmi,ki,kpi,kp->km", u, d, u, b, optimize="optimal"
        )  # type:ignore[no-any-return]

    def prox_l(self, v: npt.NDArray[np.float64], rho: float) -> npt.NDArray[np.float64]:
        return self.graph.laplacian_prox(v, rho)
