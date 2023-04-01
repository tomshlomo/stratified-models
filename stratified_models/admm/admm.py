from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from stratified_models.scalar_function import Array, ProxableScalarFunction


@dataclass
class ConsensusProblem:
    f: list[tuple[ProxableScalarFunction[Array], float]]
    g: ProxableScalarFunction[Array]
    # m: int

    @property
    def n(self) -> int:
        return len(self.f)

    def cost(self, x: Array) -> float:
        return sum(ff(x) * gamma for ff, gamma in self.f) + self.g(x)


@dataclass
class ADMMState:
    x: Array
    u: Array
    z: Array
    t: float
    primal_residual_norm: float
    dual_residual_norm: float


def norm(x: Array) -> float:
    return math.sqrt(float(x.flatten() @ x.flatten()))


@dataclass
class ConsensusADMMSolver:
    max_iterations: int = 1000
    eps_abs: float = 1e-5
    eps_rel: float = 1e-4
    mu: float = 10.0
    tau_incr: float = 2.0
    tau_decr: float = 1 / 2

    def solve(
        self,
        problem: ConsensusProblem,
        x0: Array,
        y0: Array,
        t0: float = 1.0,
    ) -> tuple[Array, float, ADMMState]:
        assert problem.n == y0.shape[0]
        state = self.get_init_state(x0=x0, y0=y0, t0=t0, problem=problem)
        costs = [[problem.cost(x) for x in state.x] + [problem.cost(state.z)]]
        start = time.time()
        for i in tqdm(range(self.max_iterations), total=self.max_iterations):
            state = self.step(problem, state)
            costs.append([problem.cost(x) for x in state.x] + [problem.cost(state.z)])
            print(f"{i} {time.time() - start:.2f} {min(costs[-1])}")
            if self._stop(state=state, problem=problem):
                break
        return state.z, costs[-1][-1], state

    def step(
        self,
        problem: ConsensusProblem,
        state: ADMMState,
    ) -> ADMMState:
        # t update
        t, u = self._t_update(problem=problem, state=state)

        # x update
        new_x = np.array(
            [
                f.prox(state.z - u, t * gamma)
                for (f, gamma), x, u in zip(problem.f, state.x, u)
            ]
        )

        # z update
        u_bar = np.mean(u, axis=0)
        new_x_bar = np.mean(new_x, axis=0)
        new_z = problem.g.prox(new_x_bar + u_bar, t / problem.n)

        # u update
        primal_residual = new_x - new_z
        new_u = u + primal_residual

        dual_residual_norm = norm(state.z - new_z) * math.sqrt(problem.n) / t

        # todo: t update
        return ADMMState(
            x=new_x,
            z=new_z,
            u=new_u,
            t=t,
            dual_residual_norm=dual_residual_norm,
            primal_residual_norm=norm(primal_residual),
        )

    def _stop(self, state: ADMMState, problem: ConsensusProblem) -> bool:
        # n is the size of each x
        # m is the size of z
        # p is the size of c
        # in my termns, assuming z.size = m, we have:
        # n = m * n
        # m = n
        # p = m * n
        eps_abs = math.sqrt(state.x.size) * self.eps_abs

        # dual residual norm
        # norm(y) = norm(u*rho)=norm(u/t) = norm(u) / sqrt(t)
        eps_rel_dual = self.eps_rel * norm(state.u) / math.sqrt(state.t)
        eps_dual = eps_abs + eps_rel_dual
        if state.dual_residual_norm > eps_dual:
            return False

        # primal residual norm
        eps_rel_primal = self.eps_rel * max(
            [
                norm(state.x),
                norm(state.z) * math.sqrt(problem.n),
            ]
        )
        eps_primal = eps_abs + eps_rel_primal
        return state.primal_residual_norm <= eps_primal

    def _t_update(
        self, state: ADMMState, problem: ConsensusProblem
    ) -> tuple[float, Array]:
        if state.primal_residual_norm >= self.mu * state.dual_residual_norm:
            return state.t * self.tau_decr, state.u * self.tau_decr
        if state.dual_residual_norm >= self.mu * state.primal_residual_norm:
            return state.t * self.tau_incr, state.u * self.tau_incr
        return state.t, state.u

    def get_init_state(
        self,
        x0: Array,
        y0: Array,
        t0: float,
        problem: ConsensusProblem,
    ) -> ADMMState:
        return ADMMState(
            x=np.array([x0] * problem.n),
            z=x0,
            u=y0 * t0,
            t=t0,
            dual_residual_norm=np.nan,
            primal_residual_norm=np.nan,
        )
