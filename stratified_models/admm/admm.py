from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from stratified_models.scalar_function import Array, ProxableScalarFunction


@dataclass
class ConsensusProblem:
    f: list[tuple[ProxableScalarFunction[Array], float]]
    g: ProxableScalarFunction[Array]
    var_shape: tuple[int, ...]
    # m: int

    @property
    def n(self) -> int:
        return len(self.f)

    def cost(self, x: Array) -> float:
        return sum(ff(x) * gamma for ff, gamma in self.f) + self.g(x)


@dataclass
class ADMMState:
    u: Array
    z: Array
    t: Array

    @property
    def rho(self) -> Array:
        return 1 / self.t

    # def y(self) -> Array:
    #     return np.einsum("i...,i->i...", self.u, self.rho)

    def y_norm(self) -> float:
        return float(np.sqrt(np.sum(norms_squared(self.u) / self.t)))


@dataclass
class ADMMStateWithStopInfo(ADMMState):
    primal_residuals_norms: Array
    dual_residuals_norms: Array
    x_norm: float

    @property
    def primal_residual_norm(self) -> float:
        return norm(self.primal_residuals_norms)

    @property
    def dual_residual_norm(self) -> float:
        return norm(self.dual_residuals_norms)

    # def total_norm(self) -> float:
    # return math.sqrt(self.dual_residual_norm**2 + self.primal_residual_norm**2)


def norm_squared(x: Array) -> float:
    xf = x.flatten()
    return float(xf @ xf)


def norm(x: Array) -> float:
    return math.sqrt(norm_squared(x))


def norms_squared(x: Array) -> Array:
    xf = x.reshape([x.shape[0], -1])
    return np.einsum("ij,ij->i", xf, xf)  # type:ignore[no-any-return]


def norms(x: Array) -> Array:
    return np.sqrt(norms_squared(x))


@dataclass
class ConsensusADMMSolver:
    max_iterations: int = 1000
    eps_abs: float = 1e-6
    eps_rel: float = 1e-4
    mu: float = 10.0
    tau_incr: float = 2.0
    tau_decr: float = 1 / 2
    k: int = 20  # todo: rename
    tau: float = 5  # todo: rename

    def solve(
        self,
        problem: ConsensusProblem,
        initial_state_candidates: Optional[list[ADMMState]] = None,
    ) -> tuple[Array, float, ADMMState]:
        initial_state_candidates = initial_state_candidates or []
        state = self._get_init_state(
            initial_state_candidates=initial_state_candidates,
            problem=problem,
        )
        best_cost = problem.cost(state.z)
        best_z = state.z
        costs_vec = np.empty(self.max_iterations)
        print("Starting ADMM iterations")
        print(
            f"{'iteration':10s} "
            f"{'time (sec)':10s} "
            f"{'cost':10s} "
            f"{'prim_res':10s} "
            f"{'dual_res':10s}"
        )
        print(" ".join(["=" * 10] * 5))
        start = time.time()
        for i in range(self.max_iterations):
            state_tmp = self._t_update(state=state)
            state = self._step(problem, state_tmp)
            cost = problem.cost(state.z)
            costs_vec[i] = cost
            if cost <= best_cost:
                best_cost = cost
                best_z = state.z

            print(
                f"{i:10d} "
                f"{time.time() - start:10.2f} "
                f"{cost:10.3e} "
                f"{state.primal_residual_norm:10.3e} "
                f"{state.dual_residual_norm:10.3e}"
            )
            if self._stop(
                state=state,
                problem=problem,
                costs=costs_vec,
                i=i,
            ):
                break
        return best_z, best_cost, state

    def _step(
        self,
        problem: ConsensusProblem,
        state: ADMMState,
    ) -> ADMMStateWithStopInfo:
        t, u, z = state.t, state.u, state.z
        rho = 1 / t
        total_rho = float(np.sum(rho))
        w = rho / total_rho

        # x update
        new_x = np.array(
            [
                f.prox(z - uu, tt * gamma)
                for (f, gamma), uu, tt in zip(
                    problem.f, u, t
                )  # todo: we don't need x in the state
            ]
        )

        # z update
        u_bar = np.einsum("i,i...->...", w, u)
        new_x_bar = np.einsum("i,i...->...", w, new_x)
        new_z = problem.g.prox(new_x_bar + u_bar, 1 / total_rho)

        # u update
        primal_residual = new_x - new_z
        new_u = u + primal_residual

        primal_residual_norms = norms(primal_residual)
        dual_residual_norms = norm(state.z - new_z) * rho
        return ADMMStateWithStopInfo(
            z=new_z,
            u=new_u,
            t=t,
            primal_residuals_norms=primal_residual_norms,
            dual_residuals_norms=dual_residual_norms,
            x_norm=norm(new_x),
        )

    def _stop(
        self,
        state: ADMMStateWithStopInfo,
        problem: ConsensusProblem,
        costs: Array,
        i: int,
    ) -> bool:
        if i <= self.k:
            return False
        eps_abs = math.sqrt(state.u.size) * self.eps_abs

        # dual residual norm
        eps_rel_dual = self.eps_rel * state.y_norm()
        eps_dual = eps_abs + eps_rel_dual
        if state.dual_residual_norm > eps_dual:
            return False

        # primal residual norm
        eps_rel_primal = self.eps_rel * max(
            [
                state.x_norm,
                norm(state.z) * math.sqrt(problem.n),
            ]
        )
        eps_primal = eps_abs + eps_rel_primal
        if state.primal_residual_norm > eps_primal:
            return False

        p = self._estimate_optimal_value(costs[: (i + 1)])
        return costs[i] - p <= self.eps_rel * p  # type:ignore[no-any-return]

    def _estimate_optimal_value(self, costs: Array) -> float:
        """
        sum(cost[j] - cost[j+1] j=i...) = cost[i] - p*
        where p* is the optimal value.
        let:
        x[i] = log(cost[i] - cost[i-1])
        and assume we have a and b such that:
        x[i] = a + b*i
        (which is equivalent to assuming cost[i] = p* + cost0 * exp(-i/tau))

        then:
        p* = cost[i] - sum(exp(x[j]) for j=i..)
           = cost[i] - sum(exp(a + b*j) for j=i..)
           = cost[i] - exp(a + bi) / (1 - exp(b))
        for convenience we will always reindex such that i=0
        """
        # todo: improve speed:
        #  rank1 updates to least squares? or at least to sums and sos
        #  cache log of diffs in in the state itself.
        y = np.log(-np.diff(costs[-(self.k + 1) :]))
        mask = np.isnan(y)
        y[mask] = 0.0
        x = np.arange(-self.k + 1, 1)
        w = 2.0 ** (x / self.tau)
        w[mask] = 0.0
        # todo: use cov=True, and estimate a lower bound on p
        #  using the covariance + delta method? might be an overkill
        z = np.polyfit(x=x, y=y, deg=1, w=w)
        b, a = z
        delta = np.exp(a) / (1 - np.exp(b))
        return float(costs[-1] - delta)

    def _t_update(self, state: ADMMStateWithStopInfo) -> ADMMState:
        mask_decr = state.primal_residuals_norms >= self.mu * state.dual_residuals_norms
        mask_incr = state.dual_residuals_norms >= self.mu * state.primal_residuals_norms
        if not np.any(mask_decr) and not np.any(mask_incr):
            return state

        factors = np.ones(mask_incr.shape)
        factors[mask_decr] = self.tau_decr
        factors[mask_incr] = self.tau_incr

        t = state.t * factors
        u = np.einsum("i...,i->i...", state.u, factors)
        return ADMMState(
            u=u,
            z=state.z,
            t=t,
        )

    def _get_init_state(
        self,
        problem: ConsensusProblem,
        initial_state_candidates: list[ADMMState],
    ) -> ADMMStateWithStopInfo:
        zero_state = self._get_zero_state(problem=problem)
        initial_state_candidates.append(zero_state)

        # advance one step for each state
        next_states = map(
            lambda state: self._step(problem, state), initial_state_candidates
        )

        # return state with the lowest cost
        return min(next_states, key=lambda state: problem.cost(state.z))

    def _get_zero_state(self, problem: ConsensusProblem) -> ADMMState:
        return ADMMState(
            z=np.zeros(problem.var_shape),
            u=np.zeros((problem.n, *problem.var_shape)),
            t=np.ones(problem.n),
        )
