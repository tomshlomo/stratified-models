from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from stratified_models.scalar_function import ScalarFunction

# Var = npt.NDArray[np.float64]
# Vars = npt.NDArray[np.float64]
# DualVars = npt.NDArray[np.float64]
T = TypeVar("T")


class Proxable(ScalarFunction[T]):
    def prox(self, v: T, t: float) -> T:
        pass


@dataclass
class ConsensusProblem(Generic[T]):
    f: list[tuple[Proxable[T], float]]
    g: Proxable[T]
    # m: int

    @property
    def n(self):
        return len(self.f)

    def cost(self, x: T) -> float:
        return sum(ff(x) * gamma for ff, gamma in self.f) + self.g(x)


@dataclass
class ADMMState(Generic[T]):
    x: npt.NDArray[T]
    u: npt.NDArray[T]
    z: T
    t: float

    @property
    def n(self) -> int:
        return len(self.x)


class ConsensusADMMSolver(Generic[T]):
    def solve(
        self,
        problem: ConsensusProblem,
        x0: T,
        y0: npt.NDArray[T],
        t0: float = 1.0,
    ) -> tuple[T, float, ADMMState[T]]:
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
        x0: T,
        y0: npt.NDArray[T],
        t0: float,
        problem: ConsensusProblem,
    ) -> ADMMState:
        return ADMMState(
            x=np.array([x0] * problem.n),
            z=x0,
            u=y0 * t0,
            t=t0,
        )
