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
    x: Array
    u: Array
    z: Array
    t: Array
    primal_residual_norm: float
    dual_residual_norm: float
    primal_residual_norms: list[float] = None

    @property
    def rho(self) -> Array:
        return 1 / self.t

    def y(self) -> Array:
        return np.einsum("i...,i->i...", self.u, self.rho)

    def total_norm(self) -> float:
        return math.sqrt(self.dual_residual_norm**2 + self.primal_residual_norm**2)


def norm(x: Array) -> float:
    return math.sqrt(float(x.flatten() @ x.flatten()))


@dataclass
class ConsensusADMMSolver:
    max_iterations: int = 1000
    eps_abs: float = 1e-9
    eps_rel: float = 1e-9
    mu: float = 10.0
    tau_incr: float = 2.0
    tau_decr: float = 1 / 2

    def solve(
        self,
        problem: ConsensusProblem,
        initial_state_candidates: Optional[list[ADMMState]] = None,
    ) -> tuple[Array, float, ADMMState]:
        initial_state_candidates = initial_state_candidates or []
        state = self._get_init_state(
            initial_state_candidates=initial_state_candidates, problem=problem
        )
        costs = [[problem.cost(x) for x in state.x] + [problem.cost(state.z)]]
        start = time.time()
        residuals = []
        for i in range(self.max_iterations):
            state = self.step(problem, state)
            costs.append([problem.cost(x) for x in state.x] + [problem.cost(state.z)])
            residuals.append(
                state.primal_residual_norms
                + [state.primal_residual_norm, state.dual_residual_norm]
            )
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
        rho = 1 / t
        total_rho = float(np.sum(rho))
        w = rho / total_rho

        # x update
        new_x = np.array(
            [
                f.prox(state.z - uu, tt * gamma)
                for (f, gamma), x, uu, tt in zip(problem.f, state.x, u, t)
            ]
        )

        # z update
        u_bar = np.einsum("i,i...->...", w, u)
        new_x_bar = np.einsum("i,i...->...", w, new_x)
        new_z = problem.g.prox(new_x_bar + u_bar, 1 / total_rho)

        # u update
        primal_residual = new_x - new_z
        new_u = u + primal_residual

        dual_residual_norm = norm(state.z - new_z) * np.sqrt(
            np.sum(rho**2)
        )  # todo: use norm squared

        # todo: t update
        return ADMMState(
            x=new_x,
            z=new_z,
            u=new_u,
            t=t,
            dual_residual_norm=dual_residual_norm,
            primal_residual_norm=norm(primal_residual),
            primal_residual_norms=[norm(r) for r in primal_residual],
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
        eps_rel_dual = self.eps_rel * norm(
            state.y()
        )  # todo: not efficient  - divide after norm, not before
        # eps_rel_dual = self.eps_rel * norm(state.u) / math.sqrt(state.t)
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
    ) -> tuple[Array, Array]:
        # if state.primal_residual_norm >= self.mu * state.dual_residual_norm:
        #     return state.t * self.tau_decr, state.u * self.tau_decr
        # if state.dual_residual_norm >= self.mu * state.primal_residual_norm:
        #     return state.t * self.tau_incr, state.u * self.tau_incr
        return state.t, state.u

    def _get_init_state(
        self,
        problem: ConsensusProblem,
        initial_state_candidates: list[ADMMState],
    ) -> ADMMState:
        """
        argmin 1/2 x'Qx + 1/2t (x - z + u) = z
        Qx + 1/t (x - z + u) = 0
        (Q+I/t)x = 1/t(z-u)
        x = (Q+I/t)^-1 1/t (z-u) = z
        ((tQ+I)^-1 - I)z = (tQ+I)^-1 u
        u = (tQ+I)z

        grad f (x) + rho (x - z + u) = 0
        (grad f + rho I) x = rho z - rho u
        what u should be to get prox(z-u)=z?
        (grad f + rho I) z = rho z - rho u
        u = -grad f * t
        """

        zero_state = self._get_zero_state(problem=problem)
        if not initial_state_candidates:
            return zero_state
        initial_state_candidates.append(zero_state)
        if initial_state_candidates:
            return min(
                initial_state_candidates, key=lambda state: problem.cost(state.z)
            )
        state = min(initial_state_candidates, key=lambda state: problem.cost(state.z))
        # state = zero_state
        z = state.z
        states = []
        for u in [
            # np.array([-f.grad(z) * gamma * state.t for f, gamma in problem.f]),
            np.zeros(state.u.shape),
            # state.u,
        ]:
            new_state = ADMMState(
                x=np.concatenate([[z]] * problem.n),
                z=z,
                u=u,
                t=state.t,
                primal_residual_norm=np.nan,
                dual_residual_norm=np.nan,
            )
            states.append(self.step(problem, new_state))
        state = min(states, key=lambda state: state.dual_residual_norm)
        return state
        # def key(state: ADMMState) -> float:
        #     next_state = self.step(problem, state)
        #     print(next_state.total_norm())
        #     return next_state.total_norm()
        # # next_states = (self.step(problem, state)
        # for state in initial_state_candidates)
        # # state = min(initial_state_candidates, key=lambda )
        #
        # # todo: nan the residual norms
        # state = min(initial_state_candidates, key=lambda state: problem.cost(state.z))
        # z = state.z
        # u = np.array([-f.grad(z) * gamma * state.t for f, gamma in problem.f])
        # return ADMMState(
        #     x=np.concatenate([[z]] * problem.n),
        #     z=z,
        #     u=u,
        #     t=state.t,
        #     primal_residual_norm=np.nan,
        #     dual_residual_norm=np.nan,
        # )

    def _get_zero_state(self, problem: ConsensusProblem):
        return ADMMState(
            x=np.zeros((problem.n, *problem.var_shape)),
            z=np.zeros(problem.var_shape),
            u=np.zeros((problem.n, *problem.var_shape)),
            t=np.ones(problem.n),
            # t=1000.0,
            dual_residual_norm=np.nan,
            primal_residual_norm=np.nan,
        )
