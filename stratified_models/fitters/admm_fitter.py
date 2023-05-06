from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable

import numpy as np
import pandas as pd
import scipy

from stratified_models.admm.admm import ADMMState, ConsensusADMMSolver, ConsensusProblem
from stratified_models.fitters.fitter import Fitter, ProblemUpdate, RefitDataBase
from stratified_models.problem import StratifiedLinearRegressionProblem, Theta
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
class ADMMRefitData(RefitDataBase):
    previous_final_states: list[ADMMState]
    previous_admm_problem: ConsensusProblem


@dataclass
class ADMMFitter(Fitter[ProxableScalarFunction[Array], ADMMRefitData]):
    solver: ConsensusADMMSolver = field(default_factory=ConsensusADMMSolver)
    max_refit_data_size: int = 10  # todo: increase

    @staticmethod
    def _build_theta_df(
        theta: Array,
        problem: StratifiedLinearRegressionProblem,
    ) -> pd.DataFrame:
        return pd.DataFrame(  # todo: should be a common function
            theta.reshape(problem.theta_flat_shape()),
            index=pd.MultiIndex.from_product(
                graph.nodes for graph, _ in problem.graphs
            ),
            columns=problem.regression_features,
        )

    def refit(
        self,
        problem_update: ProblemUpdate,
        refit_data: ADMMRefitData,
    ) -> tuple[Theta, ADMMRefitData, float]:
        admm_problem = self._build_admm_problem_from_previous(
            problem_update=problem_update,
            refit_data=refit_data,
        )
        theta, cost, final_admm_state = self.solver.solve(
            problem=admm_problem,
            initial_state_candidates=refit_data.previous_final_states,
        )

        theta_df = self._build_theta_df(
            theta=theta, problem=refit_data.previous_problem
        )
        refit_data_out = self._update_refit_data(
            refit_data_in=refit_data,
            final_admm_state=final_admm_state,
        )
        return (
            Theta(df=theta_df, shape=refit_data.previous_problem.theta_shape()),
            refit_data_out,
            cost,
        )

    def _update_refit_data(
        self,
        refit_data_in: ADMMRefitData,
        final_admm_state: ADMMState,
    ) -> ADMMRefitData:
        previous_final_states = [final_admm_state] + refit_data_in.previous_final_states
        if len(previous_final_states) > self.max_refit_data_size:
            previous_final_states = previous_final_states[: self.max_refit_data_size]
        return ADMMRefitData(
            previous_final_states=previous_final_states,
            previous_admm_problem=refit_data_in.previous_admm_problem,
            previous_problem=refit_data_in.previous_problem,
        )

    def fit(
        self,
        problem: StratifiedLinearRegressionProblem[ProxableScalarFunction[Array]],
    ) -> tuple[Theta, ADMMRefitData, float]:
        admm_problem = self._build_admm_problem_from_scratch(problem)
        theta, cost, final_admm_state = self.solver.solve(
            problem=admm_problem,
        )
        theta_df = self._build_theta_df(theta=theta, problem=problem)
        refit_data = ADMMRefitData(
            previous_final_states=[final_admm_state],
            previous_admm_problem=admm_problem,
            previous_problem=problem,
        )
        return Theta(df=theta_df, shape=problem.theta_shape()), refit_data, cost

    @staticmethod
    def _build_admm_problem_from_previous(
        problem_update: ProblemUpdate,
        refit_data: ADMMRefitData,
    ) -> ConsensusProblem:
        previous_admm_problem = refit_data.previous_admm_problem
        gammas = (
            [1.0]
            + problem_update.new_regularization_gammas
            + problem_update.new_graph_gammas
        )
        f = [
            (func, gamma_new)
            for gamma_new, (func, gamma_old) in zip(gammas, previous_admm_problem.f)
        ]
        g = previous_admm_problem.g
        return ConsensusProblem(f=f, g=g, var_shape=previous_admm_problem.var_shape)

    def _build_admm_problem_from_scratch(
        self,
        problem: StratifiedLinearRegressionProblem[ProxableScalarFunction[Array]],
    ) -> ConsensusProblem:
        f: list[tuple[ProxableScalarFunction[Array], float]] = [
            (self._get_loss(problem), 1.0)
        ]
        f.extend(problem.regularizers)
        f.extend(problem.laplacians())
        return ConsensusProblem(
            f=f,
            g=Zero(),  # todo: enforce domain constraints?
            var_shape=problem.theta_shape(),
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

    def grad(self, x: Array) -> Array:
        g = np.zeros(x.shape)
        for loss_data in self.losses:
            g_local = loss_data.loss.grad(
                self._concatenate_variables_cluster(x, loss_data),
            )
            g_local = g_local.reshape((loss_data.size, -1))
            for i, g_node in zip(loss_data.nodes_indices, g_local):
                g[i] = g_node
        return x
