from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable

import numpy as np
import pandas as pd
import scipy

from stratified_models.admm.admm import ConsensusADMMSolver, ConsensusProblem
from stratified_models.fitters.fitter import Fitter, Theta
from stratified_models.problem import StratifiedLinearRegressionProblem
from stratified_models.scalar_function import Array, ProxableScalarFunction, Zero

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
class ADMMFitter(Fitter[ProxableScalarFunction[Array]]):
    solver: ConsensusADMMSolver = field(default_factory=ConsensusADMMSolver)

    def fit(
        self,
        problem: StratifiedLinearRegressionProblem[ProxableScalarFunction[Array]],
    ) -> Theta:
        admm_problem = self._build_admm_problem(problem)
        theta, cost, final_admm_state = self.solver.solve(
            admm_problem,
            x0=np.zeros(problem.theta_shape()),
            y0=np.zeros((admm_problem.n, *problem.theta_shape())),
            t0=1.0,  # todo: smart t0
        )
        theta_df = pd.DataFrame(  # todo: should be a common function
            theta.reshape(problem.theta_flat_shape()),
            index=pd.MultiIndex.from_product(
                graph.nodes for graph, _ in problem.graphs
            ),
            columns=problem.regression_features,
        )
        return Theta(df=theta_df)

    def _build_admm_problem(
        self, problem: StratifiedLinearRegressionProblem[ProxableScalarFunction[Array]]
    ) -> ConsensusProblem:
        # problem.
        # k, m = problem.theta_flat_shape()
        # km = k * m
        f: list[tuple[ProxableScalarFunction[Array], float]] = [
            (self._get_loss(problem), 1.0)
        ]
        f.extend(problem.regularizers)
        f.extend(problem.laplacians())
        # f.extend(self._get_laplacian_regularizers(problem))
        return ConsensusProblem(
            f=f,
            g=Zero(),  # enforce domain constraints?
        )

    def _get_node_clusters(
        self, problem: StratifiedLinearRegressionProblem
    ) -> Iterable[NodesCluster]:
        rows_so_far = 0
        cols_so_far = 0
        max_rank = max(1024, problem.m)
        cluster = NodesCluster.empty()
        for node, x_slice in problem.x.groupby(problem.stratification_features):
            size = x_slice.shape[0]
            if min(rows_so_far + size, cols_so_far + problem.m) > max_rank:
                yield cluster
                rows_so_far = 0
                cols_so_far = 0
                cluster = NodesCluster.empty()
            rows_so_far += size
            cols_so_far += problem.m
            cluster.nodes.append(node)
            cluster.x_slices.append(x_slice[problem.regression_features].values)
            cluster.y_slices.append(problem.y[x_slice.index].values)
        if not cluster.is_empty:
            yield cluster

    def _get_loss(
        self, problem: StratifiedLinearRegressionProblem[ProxableScalarFunction[Array]]
    ) -> LossForADMM:
        losses = []
        for nodes_cluster in self._get_node_clusters(problem):
            loss = problem.loss_factory.build_loss_function(
                x=scipy.sparse.block_diag(nodes_cluster.x_slices),
                y=np.concatenate(nodes_cluster.y_slices),
            )
            losses.append(
                NodesClusterLossData(
                    nodes_indices=[
                        problem.get_node_index(node) for node in nodes_cluster.nodes
                    ],
                    loss=loss,
                )
            )
        return LossForADMM(losses=losses)

    # def _get_laplacian_regularizers(
    #         self,
    #         problem: StratifiedLinearRegressionProblem[ProxableScalarFunction[Array]],
    # ) -> list[tuple[ProxableScalarFunction[Array][Theta], float]]:
    #     problem.laplacians()


@dataclass
class NodesCluster:
    nodes: list[Hashable | tuple[Hashable, ...]]
    x_slices: list[Array]
    y_slices: list[Array]

    @classmethod
    def empty(cls):
        return NodesCluster(
            nodes=[],
            x_slices=[],
            y_slices=[],
        )

    @property
    def is_empty(self) -> bool:
        return not self.nodes


@dataclass
class NodesClusterLossData:
    nodes_indices: list[tuple[int, ...]]
    loss: ProxableScalarFunction[Array]

    @property
    def size(self) -> int:
        return len(self.nodes_indices)


@dataclass
class LossForADMM(ProxableScalarFunction[Array]):
    losses: list[NodesClusterLossData]

    @staticmethod
    def _concatenate_variables_cluster(
        x: Array, cluster: NodesClusterLossData
    ) -> Array:
        return np.concatenate([x[index] for index in cluster.nodes_indices])

    def __call__(self, x: Array) -> float:
        return sum(
            loss_data.loss(self._concatenate_variables_cluster(x, loss_data))
            for loss_data in self.losses
        )

    def prox(self, v: Array, t: float) -> Array:
        x = v.copy()
        for loss_data in self.losses:
            x_local = loss_data.loss.prox(
                v=self._concatenate_variables_cluster(v, loss_data),
                t=t,
            )
            x_local = x_local.reshape((loss_data.size, -1))
            for i, x_node in zip(loss_data.nodes_indices, x_local):
                x[i] = x_node
        return x
