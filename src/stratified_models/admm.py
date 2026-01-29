from __future__ import annotations

import math
import time
from dataclasses import dataclass

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import structlog

from stratified_models.problem import SolveInfo
from stratified_models.scalar_function import (
    Proxable,
)

logger = structlog.get_logger()


@dataclass
class ConsensusProblem:
    f: tuple[tuple[Proxable, float], ...]
    g: Proxable
    var_shape: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.f)

    def cost(self, x: jax.Array) -> float:
        val = 0.0
        for ff, gamma in self.f:
            val += float(ff(x)) * gamma
        val += float(self.g(x))
        return val


@dataclass
class ADMMState(SolveInfo):
    u: jax.Array
    z: jax.Array
    t: jax.Array
    primal_residuals_norms: jax.Array | None = None
    dual_residuals_norms: jax.Array | None = None
    x_norm: jax.Array | None = None
    converged_: bool = False

    @property
    def rho(self) -> jax.Array:
        return 1 / self.t

    def y_norm(self) -> jax.Array:
        u_norms_squared = norm_squared(self.u, axis=tuple(range(1, self.u.ndim)))
        return jnp.sqrt(jnp.sum(u_norms_squared / self.t))

    @property
    def primal_residual_norm(self) -> jax.Array:
        assert self.primal_residuals_norms is not None
        return jnp.linalg.norm(self.primal_residuals_norms)

    @property
    def dual_residual_norm(self) -> jax.Array:
        assert self.dual_residuals_norms is not None
        return jnp.linalg.norm(self.dual_residuals_norms)

    def converged(self) -> bool:
        return self.converged_


jax.tree_util.register_pytree_node(
    ADMMState,
    lambda state: (
        (
            state.u,
            state.z,
            state.t,
            state.primal_residuals_norms,
            state.dual_residuals_norms,
            state.x_norm,
        ),
        (state.converged_,),
    ),
    lambda aux, children: ADMMState(
        u=children[0],
        z=children[1],
        t=children[2],
        primal_residuals_norms=children[3],
        dual_residuals_norms=children[4],
        x_norm=children[5],
        converged_=aux[0],
    ),
)


def norm_squared(
    x: jax.Array, *, axis: int | tuple[int, ...] | None = None
) -> jax.Array:
    return jnp.sum(jnp.square(x), axis=axis)


@attrs.frozen(kw_only=True)
class ConsensusADMM:
    max_iterations: int = 1000
    eps_abs: float = 1e-6
    eps_rel: float = 1e-4
    mu: float = 10.0
    tau_incr: float = 2.0
    tau_decr: float = 1 / 2
    k: int = 20
    tau: float = 5

    def solve(
        self,
        problem: ConsensusProblem,
        initial_state: ADMMState | None = None,
    ) -> tuple[jax.Array, float, ADMMState]:
        state = initial_state or self._get_zero_state(problem)
        best_cost = problem.cost(state.z)
        best_z = state.z
        costs_vec = []

        start_time = time.time()
        converged = False

        @jax.jit
        def step_fn(state: ADMMState) -> ADMMState:
            return self._step(problem, state)

        @jax.jit
        def t_update_fn(state: ADMMState) -> ADMMState:
            return self._t_update(state)

        # We use a loop instead of lax.scan because of the convergence check
        # and the heterogeneous nature of f (can't vmap prox easily if types differ)
        for i in range(self.max_iterations):
            state = t_update_fn(state)
            state = step_fn(state)
            cost = problem.cost(state.z)
            costs_vec.append(cost)

            if cost <= best_cost:
                best_cost = cost
                best_z = state.z

            if self._stop(state, problem, costs_vec, i):
                converged = True
                break

        state.converged_ = converged
        logger.info(
            "admm_solve_complete",
            iterations=len(costs_vec),
            best_cost=best_cost,
            converged=converged,
            duration=time.time() - start_time,
        )
        return best_z, best_cost, state

    def _step(self, problem: ConsensusProblem, state: ADMMState) -> ADMMState:
        t, u, z, rho = state.t, state.u, state.z, state.rho
        total_rho = jnp.sum(rho)
        w = rho / total_rho

        # x update
        # We iterate over f because they are different objects
        new_x_list = []
        for _i, ((f, gamma), uu, tt) in enumerate(zip(problem.f, u, t, strict=True)):
            # prox_t f(v)
            # here we want argmin f_k(x) + rho/2 |x - z + u|^2
            # = prox_{1/rho} f_k (z - u)
            # = prox_{t} f_k (z - u)
            # Wait, the legacy code had: f.prox(z - uu, tt * gamma)
            # Legacy: f_k(theta) * gamma.
            # prox_{t*gamma} f_k (z - u)
            # Yes, if we minimize gamma * f(x) + 1/(2t) |x-v|^2
            # it is equivalent to minimizing f(x) + 1/(2t*gamma) |x-v|^2
            # so the parameter passed to prox of f is t*gamma.
            val = f.prox(z - uu, tt * gamma)
            new_x_list.append(val)

        new_x = jnp.stack(new_x_list)

        # z update
        # z = prox_g ( sum(w_i * (x_i + u_i)) )
        # with parameter 1/total_rho
        u_bar = jnp.einsum("i,i...->...", w, u)
        new_x_bar = jnp.einsum("i,i...->...", w, new_x)
        new_z = problem.g.prox(new_x_bar + u_bar, 1 / total_rho)

        # u update
        primal_residual = new_x - new_z
        new_u = u + primal_residual

        primal_residual_norms = jnp.linalg.vector_norm(
            primal_residual, axis=tuple(range(1, primal_residual.ndim))
        )
        dual_residual_norms = jnp.linalg.norm(state.z - new_z) * rho

        return ADMMState(
            z=new_z,
            u=new_u,
            t=t,
            primal_residuals_norms=primal_residual_norms,
            dual_residuals_norms=dual_residual_norms,
            x_norm=jnp.linalg.norm(new_x),
        )

    def _stop(
        self,
        state: ADMMState,
        problem: ConsensusProblem,
        costs: list[float],
        i: int,
    ) -> bool:
        if i <= self.k:
            return False

        assert state.primal_residuals_norms is not None
        assert state.dual_residuals_norms is not None
        assert state.x_norm is not None

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
                jnp.linalg.norm(state.z) * math.sqrt(problem.n),
            ],
        )
        eps_primal = eps_abs + eps_rel_primal
        if state.primal_residual_norm > eps_primal:
            return False

        p = self._estimate_optimal_value(costs[: (i + 1)])
        return costs[i] - p <= self.eps_rel * abs(p)

    def _estimate_optimal_value(self, costs: list[float]) -> float:
        # Same as legacy
        costs_arr = np.array(costs)
        y = np.log(-np.diff(costs_arr[-(self.k + 1) :]))
        mask = np.isnan(y)
        y[mask] = 0.0
        x = np.arange(-self.k + 1, 1)
        w = 2.0 ** (x / self.tau)
        w[mask] = 0.0

        if np.sum(w) == 0:
            return costs[-1]

        z = np.polyfit(x=x, y=y, deg=1, w=w)
        b, a = z
        if b >= 0:  # Diverging or not converging geometrically
            return costs[-1]

        delta = np.exp(a) / (1 - np.exp(b))
        return float(costs[-1] - delta)

    def _t_update(self, state: ADMMState) -> ADMMState:
        # We need to handle the case where primal_residuals_norms is None
        # But inside JIT, we can't check for None easily if it's traced.
        # However, ADMMState pytree registration handles None by passing it through?
        # No, JAX pytree leaves must be arrays. None is not an array.
        # If primal_residuals_norms is None, it will be treated as auxiliary data
        # or just None in the structure.
        # But wait, our register_pytree_node puts it in children (tuple of arrays).
        # If it is None, JAX might complain or treat it as empty node?
        # Actually, JAX allows None in pytrees.
        # But if we use it in `jnp.where`, it must be an array.
        # In the first iteration, it is None.
        # But `_t_update` is called at start of loop.
        # `state` comes from `_get_zero_state` where it is None.
        # So `_t_update` must handle None.
        # But inside JIT, we can't branch on value of None (it's static).
        # If we trace `t_update_fn` with `state` having None, it will be specialized
        # for None.
        # Then `step_fn` returns `state` with arrays.
        # Then next `t_update_fn` call will have arrays.
        # This triggers recompilation!
        # Recompilation happens once (from None to Array).
        # That is acceptable.

        if state.primal_residuals_norms is None:
            return state

        assert state.primal_residuals_norms is not None
        assert state.dual_residuals_norms is not None

        mask_decr = state.primal_residuals_norms >= self.mu * state.dual_residuals_norms
        mask_incr = state.dual_residuals_norms >= self.mu * state.primal_residuals_norms

        # Check if any update is needed
        # if not jnp.any(mask_decr) and not jnp.any(mask_incr):
        #    return state

        factors = jnp.ones(mask_incr.shape)
        factors = jnp.where(mask_decr, self.tau_decr, factors)
        factors = jnp.where(mask_incr, self.tau_incr, factors)

        t = state.t * factors
        u = jnp.einsum("i...,i->i...", state.u, factors)

        return ADMMState(
            u=u,
            z=state.z,
            t=t,
            primal_residuals_norms=state.primal_residuals_norms,
            dual_residuals_norms=state.dual_residuals_norms,
            x_norm=state.x_norm,
        )

    def _get_zero_state(self, problem: ConsensusProblem) -> ADMMState:
        return ADMMState(
            z=jnp.zeros(problem.var_shape),
            u=jnp.zeros((problem.n, *problem.var_shape)),
            t=jnp.ones(problem.n),
        )
