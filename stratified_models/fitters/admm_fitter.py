from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import numpy as np
import scipy

from stratified_models.admm.admm import ADMMState, ConsensusADMMSolver, ConsensusProblem
from stratified_models.fitters.fitter import Fitter, ProblemUpdate, RefitDataBase
from stratified_models.problem import StratifiedLinearRegressionProblem, Theta
from stratified_models.scalar_function import (
    Array,
    IntArray,
    ProxableScalarFunction,
    Zero,
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
class ADMMRefitData(RefitDataBase[ProxableScalarFunction[Array]]):
    previous_final_states: list[ADMMState]
    previous_admm_problem: ConsensusProblem


@dataclass(frozen=True)
class ADMMFitter(Fitter[ProxableScalarFunction[Array], ADMMRefitData]):
    solver: ConsensusADMMSolver = field(default_factory=ConsensusADMMSolver)
    max_refit_data_size: int = 10  # todo: increase
    max_rank: int = 1024

    def refit(
        self,
        problem_update: ProblemUpdate,
        refit_data: ADMMRefitData,
    ) -> tuple[Theta, ADMMRefitData, float]:
        admm_problem = self._build_admm_problem_from_previous(
            problem_update=problem_update,
            refit_data=refit_data,
        )
        theta, cost, final_admm_state, _ = self.solver.solve(
            problem=admm_problem,
            initial_state_candidates=refit_data.previous_final_states,
        )
        refit_data_out = self._update_refit_data(
            refit_data_in=refit_data,
            final_admm_state=final_admm_state,
        )
        return (
            Theta.from_array(arr=theta, problem=refit_data.previous_problem),
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
        theta, cost, final_admm_state, _ = self.solver.solve(
            problem=admm_problem,
        )
        refit_data = ADMMRefitData(
            previous_final_states=[final_admm_state],
            previous_admm_problem=admm_problem,
            previous_problem=problem,
        )
        return (
            Theta.from_array(arr=theta, problem=problem),
            refit_data,
            cost,
        )

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
        return ConsensusProblem(
            f=tuple(f),
            g=g,
            var_shape=previous_admm_problem.var_shape,
        )

    def _build_admm_problem_from_scratch(
        self,
        problem: StratifiedLinearRegressionProblem[ProxableScalarFunction[Array]],
    ) -> ConsensusProblem:
        f: list[tuple[ProxableScalarFunction[Array], float]] = [
            (self._get_loss(problem), 1.0)
        ]
        f.extend(problem.regularizers())
        f.extend(problem.laplacians())
        return ConsensusProblem(
            f=tuple(f),
            g=Zero(),  # todo: enforce domain constraints?
            var_shape=problem.theta_shape(),
        )

    def _get_loss(
        self,
        problem: StratifiedLinearRegressionProblem[ProxableScalarFunction[Array]],
    ) -> SeparableProxableScalarFunction:
        z = problem.x[problem.stratification_features]
        nodes, node_index = np.unique(z, axis=0, return_inverse=True)
        ravelled_nodes = np.ravel_multi_index(nodes.T, dims=problem.theta_shape()[:-1])
        num_of_nodes = nodes.shape[0]

        max_rank = max(self.max_rank, problem.m)
        max_nodes_per_cluster = min(max_rank // problem.m, num_of_nodes)
        num_of_clusters = ceil(num_of_nodes / max_nodes_per_cluster)

        shape = (problem.n, max_nodes_per_cluster * problem.m)
        data = problem.x[problem.regression_features].values.flatten()
        indices = np.add.outer(
            np.mod(node_index, max_nodes_per_cluster) * problem.m, np.arange(problem.m)
        ).flatten()
        cluster_index = node_index // max_nodes_per_cluster
        indptr = np.arange(0, (problem.n + 1) * problem.m, problem.m)
        x = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        losses = []
        for i in range(num_of_clusters):
            rows = cluster_index == i
            cluster_nodes_indices = ravelled_nodes[
                i * max_nodes_per_cluster : (i + 1) * max_nodes_per_cluster
            ]
            cluster_size = cluster_nodes_indices.shape[0]
            x_ = x[rows, :]
            if cluster_size < max_nodes_per_cluster:
                x_ = x_[:, : cluster_size * problem.m]
            cluster_loss = problem.loss_factory.build_loss_function(
                x=x_,
                y=problem.y[rows],
            )
            losses.append((cluster_nodes_indices, cluster_loss))
        return SeparableProxableScalarFunction(items=losses)


@dataclass
class SeparableProxableScalarFunction(ProxableScalarFunction[Array]):
    items: list[tuple[IntArray, ProxableScalarFunction[Array]]]

    def __call__(self, x: Array) -> float:
        x = x.reshape((-1, x.shape[-1]))
        return sum(func(x[index].flatten()) for index, func in self.items)

    def prox(self, v: Array, t: float) -> Array:
        orig_shape = v.shape
        v = v.reshape((-1, v.shape[-1]))
        x = v.copy()
        for index, func in self.items:
            x_local = func.prox(v=v[index].flatten(), t=t)
            x[index] = x_local.reshape((-1, v.shape[-1]))
            pass
        return x.reshape(orig_shape)
