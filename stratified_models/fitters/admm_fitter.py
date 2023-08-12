from __future__ import annotations

import time
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
class ADMMRefitData(RefitDataBase[ProxableScalarFunction[Array]]):
    previous_final_states: list[ADMMState]
    previous_admm_problem: ConsensusProblem


@dataclass(frozen=True)
class ADMMFitter(Fitter[ProxableScalarFunction[Array], ADMMRefitData]):
    solver: ConsensusADMMSolver = field(default_factory=ConsensusADMMSolver)
    max_refit_data_size: int = 10  # todo: increase

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

    def _get_node_clusters(
        self,
        problem: StratifiedLinearRegressionProblem[ProxableScalarFunction[Array]],
    ) -> Iterable[NodesCluster]:
        rows_so_far = 0
        cols_so_far = 0
        max_rank = max(1024, problem.m)
        cluster = NodesCluster.empty()
        for node, x_slice, y_slice in problem.node_data_iter():
            size = x_slice.shape[0]
            if min(rows_so_far + size, cols_so_far + problem.m) > max_rank:
                yield cluster
                rows_so_far = 0
                cols_so_far = 0
                cluster = NodesCluster.empty()
            rows_so_far += size
            cols_so_far += problem.m
            cluster.nodes.append(node)
            cluster.x_slices.append(x_slice.values)
            cluster.y_slices.append(y_slice.values)
        if not cluster.is_empty:
            yield cluster

    def _get_loss(
        self,
        problem: StratifiedLinearRegressionProblem[ProxableScalarFunction[Array]],
    ) -> LossForADMM:
        print("starting ADMMFitter._get_loss()")
        start = time.time()

        def process_group(group):
            pass
            return problem.loss_factory.build_loss_function(
                x=group[problem.regression_features].values,
                y=group[problem.target_column].values,
            )

        # agg = dd.Aggregation(
        #     name='loss_for_admm',
        #     chunk=lambda chunk: problem.loss_factory.build_loss_function(
        #         x=problem.df[problem.regression_features].values,
        #         y=problem.df[problem.target_column].values,
        #     ),
        #
        # )
        # problem.df.groupby(problem.stratification_features).agg(agg)
        result = (
            problem.dask_df.groupby(problem.stratification_features)
            .apply(process_group, meta=("local_loss", object))
            .compute()
        )
        for i, level in enumerate(result.index.levels):
            result.index = result.index.set_levels(level.astype(int), level=i)

        return LossForAdmmWithDask(losses=result)
        # loss(np.zeros(problem.theta_shape()))
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
        print(f"  building loss took {time.time() - start} seconds")
        return LossForADMM(losses=losses)


@dataclass
class NodesCluster:
    nodes: list[tuple[Hashable, ...]]
    x_slices: list[Array]
    y_slices: list[Array]

    @classmethod
    def empty(cls) -> NodesCluster:
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


class LossForAdmmWithDask(ProxableScalarFunction[Array]):
    def __init__(self, losses: pd.Series) -> None:
        self._losses = losses

    def __call__(self, x: Array) -> float:
        import dask.bag as db

        bag = db.from_sequence(self._losses.items()).map(lambda i_f: i_f[1](x[i_f[0]]))
        return bag.sum().compute()

    def prox(self, v: Array, t: float) -> Array:
        pass


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
