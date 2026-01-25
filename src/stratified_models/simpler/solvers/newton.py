from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from math import prod
from typing import Literal

import attrs
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from stratified_models.simpler.model import StartifiedModel
from stratified_models.simpler.solvers.types import (
    AbstractProblem,
    Hyperparameters,
    SolveInfo,
    Solver,
)


@attrs.frozen(kw_only=True)
class NewtonSolveInfo(SolveInfo):
    iterations: int
    stopping_reason: Literal[
        "grad_tol", "newton_tol", "step_too_small", "line_search_failed", "max_iters"
    ]
    final_objective_value: float
    final_grad_norm: float
    final_newton_decrement_sq: float
    last_step_alpha: float | None
    extra: Mapping[str, object] = attrs.field(factory=dict)

    def converged(self) -> bool:
        return self.stopping_reason in {"grad_tol", "newton_tol"}


@attrs.frozen(kw_only=True)
class CompiledNewtonProblem:
    problem: AbstractProblem


class PSDSolver(ABC):
    @property
    @abstractmethod
    def needs_matrix(self) -> bool:
        pass

    @abstractmethod
    def solve(
        self,
        *,
        b: jax.Array,
        matvec: Callable[[jax.Array], jax.Array],
        matrix: jax.Array | None,
        damping: float,
    ) -> jax.Array:
        pass


@attrs.frozen(kw_only=True)
class DirectPSDSolver(PSDSolver):
    @property
    def needs_matrix(self) -> bool:
        return True

    def solve(
        self,
        *,
        b: jax.Array,
        matvec: Callable[[jax.Array], jax.Array],  # noqa: ARG002
        matrix: jax.Array | None,
        damping: float,
    ) -> jax.Array:
        assert matrix is not None
        eye = jnp.eye(matrix.shape[0])
        return jnp.linalg.solve(matrix + damping * eye, b)


@attrs.frozen(kw_only=True)
class CGPSDSolver(PSDSolver):
    tol: float = 1e-8
    maxiter: int | None = None

    @property
    def needs_matrix(self) -> bool:
        return False

    def solve(
        self,
        *,
        b: jax.Array,
        matvec: Callable[[jax.Array], jax.Array],
        matrix: jax.Array | None,  # noqa: ARG002
        damping: float,
    ) -> jax.Array:
        def damped_matvec(v: jax.Array) -> jax.Array:
            return matvec(v) + damping * v

        x, _info = cg(damped_matvec, b, tol=self.tol, maxiter=self.maxiter)
        return x


@attrs.frozen(kw_only=True)
class NewtonSolver(Solver[CompiledNewtonProblem, NewtonSolveInfo]):
    max_iters: int = 50
    # Absolute tolerance for the gradient norm (fast check)
    grad_tol: float = 1e-6
    # Tolerance for the Newton Decrement (robust check)
    newton_tol: float = 1e-6
    damping: float = 0.0
    psd_solver: PSDSolver = attrs.field(factory=DirectPSDSolver)
    backtracking_steps: int = 20
    backtracking_factor: float = 0.5
    armijo: float = 1e-4

    @staticmethod
    def for_quadratic(psd_solver: PSDSolver) -> NewtonSolver:
        return NewtonSolver(
            max_iters=1,
            grad_tol=0.0,
            newton_tol=0.0,
            damping=0.0,
            psd_solver=psd_solver,
            backtracking_steps=1,
            backtracking_factor=0.5,
            armijo=0.0,
        )

    def compile(self, abstract_problem: AbstractProblem) -> CompiledNewtonProblem:
        return CompiledNewtonProblem(problem=abstract_problem)

    def solve(
        self,
        compiled_problem: CompiledNewtonProblem,
        hyperparameters: Hyperparameters,
    ) -> tuple[StartifiedModel, NewtonSolveInfo]:
        problem = compiled_problem.problem
        shape = problem.theta_shape.array_shape
        dim = int(prod(shape))

        def unpack(theta_vec: jax.Array) -> jax.Array:
            return jnp.reshape(theta_vec, shape)

        def objective_vec(theta_vec: jax.Array) -> jax.Array:
            theta = unpack(theta_vec)
            model = StartifiedModel(theta=theta, shape=problem.theta_shape)
            return problem.objectives(model).total(hyperparameters)

        grad_vec = jax.grad(objective_vec)
        hess_vec = jax.hessian(objective_vec) if self.psd_solver.needs_matrix else None

        theta_vec, info = self._solve_vec(
            dim=dim,
            objective_vec=objective_vec,
            grad_vec=grad_vec,
            hess_vec=hess_vec,
        )
        model = StartifiedModel(theta=unpack(theta_vec), shape=problem.theta_shape)
        return model, info

    def _solve_vec(
        self,
        *,
        dim: int,
        objective_vec: Callable[[jax.Array], jax.Array],
        grad_vec: Callable[[jax.Array], jax.Array],
        hess_vec: Callable[[jax.Array], jax.Array] | None,
    ) -> tuple[jax.Array, NewtonSolveInfo]:
        theta_vec = jnp.zeros((dim,))
        last_step_alpha: float | None = None
        final_newton_decrement_sq = float("nan")

        for iteration in range(1, self.max_iters + 1):
            f0 = objective_vec(theta_vec)
            g = grad_vec(theta_vec)
            g_norm = float(jnp.linalg.norm(g))

            if g_norm <= self.grad_tol:
                return theta_vec, NewtonSolveInfo(
                    iterations=iteration,
                    stopping_reason="grad_tol",
                    final_objective_value=float(f0),
                    final_grad_norm=g_norm,
                    final_newton_decrement_sq=final_newton_decrement_sq,
                    last_step_alpha=last_step_alpha,
                    extra={"converged": True},
                )

            step = self._newton_step(
                theta_vec=theta_vec,
                g=g,
                grad_vec=grad_vec,
                hess_vec=hess_vec,
            )
            newton_decrement_sq = float(g @ step)
            final_newton_decrement_sq = newton_decrement_sq

            if 0.5 * newton_decrement_sq <= self.newton_tol:
                return theta_vec, NewtonSolveInfo(
                    iterations=iteration,
                    stopping_reason="newton_tol",
                    final_objective_value=float(f0),
                    final_grad_norm=g_norm,
                    final_newton_decrement_sq=final_newton_decrement_sq,
                    last_step_alpha=last_step_alpha,
                    extra={"converged": True},
                )

            theta_next, alpha, reason = self._backtracking_update(
                theta_vec=theta_vec,
                step=step,
                f0=f0,
                newton_decrement_sq=newton_decrement_sq,
                objective_vec=objective_vec,
            )
            last_step_alpha = alpha

            if reason is not None:
                return theta_vec, NewtonSolveInfo(
                    iterations=iteration,
                    stopping_reason=reason,
                    final_objective_value=float(f0),
                    final_grad_norm=g_norm,
                    final_newton_decrement_sq=final_newton_decrement_sq,
                    last_step_alpha=last_step_alpha,
                    extra={"converged": False},
                )

            theta_vec = theta_next

        f0 = objective_vec(theta_vec)
        g_norm = float(jnp.linalg.norm(grad_vec(theta_vec)))
        return theta_vec, NewtonSolveInfo(
            iterations=self.max_iters,
            stopping_reason="max_iters",
            final_objective_value=float(f0),
            final_grad_norm=g_norm,
            final_newton_decrement_sq=final_newton_decrement_sq,
            last_step_alpha=last_step_alpha,
            extra={"converged": False},
        )

    def _newton_step(
        self,
        *,
        theta_vec: jax.Array,
        g: jax.Array,
        grad_vec: Callable[[jax.Array], jax.Array],
        hess_vec: Callable[[jax.Array], jax.Array] | None,
    ) -> jax.Array:
        theta_vec_now = theta_vec

        def matvec(v: jax.Array, theta_vec_now: jax.Array = theta_vec_now) -> jax.Array:
            return jax.jvp(grad_vec, (theta_vec_now,), (v,))[1]

        h: jax.Array | None = None
        if self.psd_solver.needs_matrix:
            assert hess_vec is not None
            h = hess_vec(theta_vec)

        return self.psd_solver.solve(
            b=g,
            matvec=matvec,
            matrix=h,
            damping=self.damping,
        )

    def _backtracking_update(
        self,
        *,
        theta_vec: jax.Array,
        step: jax.Array,
        f0: jax.Array,
        newton_decrement_sq: float,
        objective_vec: Callable[[jax.Array], jax.Array],
    ) -> tuple[
        jax.Array, float, Literal["step_too_small", "line_search_failed"] | None
    ]:
        alpha = 1.0
        for _ in range(self.backtracking_steps):
            theta_next = theta_vec - alpha * step
            if jnp.allclose(theta_vec, theta_next, atol=1e-15):
                return theta_vec, alpha, "step_too_small"

            f_next = objective_vec(theta_next)
            if float(f_next) <= float(f0) - self.armijo * alpha * newton_decrement_sq:
                return theta_next, alpha, None

            alpha *= self.backtracking_factor

        return theta_vec, alpha, "line_search_failed"
