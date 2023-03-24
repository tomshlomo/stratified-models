from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


class ConsensusADMMSolver:
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
        for i in range(1000):
            state = self.step(problem, state)
            costs.append([problem.cost(x) for x in state.x] + [problem.cost(state.z)])
            print(i, min(costs[-1]))
        return state.z, costs[-1][-1], state

    def step(
        self,
        problem: ConsensusProblem,
        state: ADMMState,
    ) -> ADMMState:
        # x update
        new_x = np.array(
            [
                f.prox(state.z - u, state.t * gamma)
                for (f, gamma), x, u in zip(problem.f, state.x, state.u)
            ]
        )

        # z update
        u_bar = np.mean(state.u, axis=0)
        new_x_bar = np.mean(new_x, axis=0)
        new_z = problem.g.prox(new_x_bar + u_bar, state.t / problem.n)

        # u update
        new_u = state.u + new_x - new_z

        # todo: t update
        return ADMMState(
            x=new_x,
            z=new_z,
            u=new_u,
            t=state.t,
        )

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
        )
