from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

Var = npt.NDArray[np.float64]
Vars = npt.NDArray[np.float64]
DualVars = npt.NDArray[np.float64]


class Proxable(Protocol):
    def eval(self, x: Var) -> float:
        pass

    def prox(self, v: Var, rho: float) -> Var:
        pass


@dataclass
class SeparableSumOfProxables:
    n: int
    m: int
    f: dict[int, Proxable]

    def eval(self, x: Var) -> float:
        x = x.reshape((self.n, self.m))
        cost = 0.0
        for i, f in self.f.items():
            cost += f.eval(x[i])
        return cost

    def prox(self, v: Var, rho: float) -> Var:
        v = v.reshape((self.n, self.m))
        for i, f in self.f.items():
            v[i] = f.prox(v[i], rho)
        return v.ravel()


@dataclass
class ConsensusProblem:
    f: list[Proxable]
    g: Proxable
    m: int

    @property
    def k(self):
        return len(self.f)

    def cost(self, x: Var) -> float:
        return sum(ff.eval(x) for ff in self.f) + self.g.eval(x)


@dataclass
class ADMMState:
    x: Vars
    z: Var
    u: DualVars
    # rho: float
    # best_cost: float
    # best_var: Var

    # def y(self) -> Var:
    #     return self.u * self.rho

    @property
    def k(self) -> int:
        return self.x.shape[0]

    @property
    def m(self) -> int:
        return self.x.shape[1]


class ConsensusADMMSolver:
    def solve(
        self, problem: ConsensusProblem, x0: Var, y0: DualVars
    ) -> tuple[Var, float]:
        assert x0.shape == (problem.m,)
        assert y0.shape == (problem.k, problem.m)

        rho = self.suggest_rho(problem, x0, y0)
        state = self.get_init_state(x0=x0, y0=y0, rho=rho, problem=problem)
        costs = [[problem.cost(x) for x in state.x] + [problem.cost(state.z)]]
        for i in range(1000):
            state = self.step(problem, state, rho=rho)
            costs.append([problem.cost(x) for x in state.x] + [problem.cost(state.z)])
            print(i, min(costs[-1]))
        return state.z, costs[-1][-1]

    def step(
        self, problem: ConsensusProblem, state: ADMMState, rho: float
    ) -> ADMMState:
        k = problem.k
        m = problem.m

        # x update
        new_x = np.zeros((k, m))
        for i, (f, x, u) in enumerate(zip(problem.f, state.x, state.u)):
            new_x[i] = f.prox(state.z - u, rho)

        # z update
        u_bar = np.mean(state.u, axis=0)
        new_x_bar = np.mean(new_x, axis=0)
        new_z = problem.g.prox(new_x_bar + u_bar, k * rho)

        # u update
        new_u = state.u + new_x - new_z

        return ADMMState(
            x=new_x,
            z=new_z,
            u=new_u,
        )

    def get_init_state(
        self, x0: Var, y0: DualVars, rho: float, problem: ConsensusProblem
    ) -> ADMMState:
        k = y0.shape[0]
        return ADMMState(
            x=np.tile(x0, (k, 1)),
            z=x0,
            u=y0 / rho,
            # rho=rho,
            # best_cost=problem.cost(x0),
            # best_var=x0,
        )

    def suggest_rho(self, problem: ConsensusProblem, x0: Var, y0: DualVars) -> float:
        return 1.0
