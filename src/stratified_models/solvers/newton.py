from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Literal

import attrs
import jax
import jax.numpy as jnp
import structlog
from jax.scipy.sparse.linalg import cg

from stratified_models.model import NodeIndex, StartifiedModel, ThetaShape
from stratified_models.scalar_function import ScalarFunction
from stratified_models.solvers.types import (
    AbstractProblem,
    Hyperparameters,
    Objectives,
    SolveInfo,
    Solver,
    compute_objectives,
)

logger = structlog.get_logger()


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

    def converged(self) -> bool:
        return self.stopping_reason in {"grad_tol", "newton_tol"}


@attrs.frozen(kw_only=True)
class CompiledNewtonProblem:
    theta_shape: ThetaShape
    losses: Sequence[tuple[NodeIndex, ScalarFunction]]
    regularizers: Mapping[str, ScalarFunction]
    laplacians: Sequence[tuple[str, ScalarFunction]]

    objective: Callable[[jax.Array, Hyperparameters], jax.Array]
    gradient: Callable[[jax.Array, Hyperparameters], jax.Array]
    hessian: Callable[[jax.Array, Hyperparameters], jax.Array] | None

    solutions_history: list[jax.Array] = attrs.field(factory=list)

    def compute_objectives(self, theta_vec: jax.Array) -> Objectives:
        theta = jnp.reshape(theta_vec, self.theta_shape.array_shape)
        return compute_objectives(
            theta, self.losses, self.regularizers, self.laplacians
        )


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
        x0: jax.Array | None,
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
        x0: jax.Array | None,  # noqa: ARG002
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
        x0: jax.Array | None,
    ) -> jax.Array:
        def damped_matvec(v: jax.Array) -> jax.Array:
            return matvec(v) + damping * v

        x, _info = cg(damped_matvec, b, x0=x0, tol=self.tol, maxiter=self.maxiter)
        return x


@attrs.frozen(kw_only=True)
class NewtonSolver(Solver[CompiledNewtonProblem, NewtonSolveInfo]):
    max_iters: int = 50
    solutions_history_size: int = 100
    # Absolute tolerance for the gradient norm (fast check)
    grad_tol: float = 1e-6
    # Tolerance for the Newton Decrement (robust check)
    newton_atol: float = 1e-4
    newton_rtol: float = 1e-4
    damping: float = 0.0
    psd_solver: PSDSolver = attrs.field(factory=DirectPSDSolver)
    backtracking_steps: int = 20
    backtracking_factor: float = 0.5
    armijo: float = 1e-4

    @staticmethod
    def for_quadratic(psd_solver: PSDSolver) -> NewtonSolver:
        return NewtonSolver(
            max_iters=2,
            damping=0.0,
            psd_solver=psd_solver,
            backtracking_steps=1,
        )

    def compile(self, abstract_problem: AbstractProblem) -> CompiledNewtonProblem:
        shape = abstract_problem.theta_shape

        losses = []
        for node, loss_fn in abstract_problem.group_losses():
            losses.append((shape.node_to_index(node), loss_fn))

        laplacians = list(abstract_problem.laplacians())
        regularizers = abstract_problem.regularizers

        def unpack(theta_vec: jax.Array) -> jax.Array:
            return jnp.reshape(theta_vec, shape.array_shape)

        def objective_fn(
            theta_vec: jax.Array, hyperparameters: Hyperparameters
        ) -> jax.Array:
            theta = unpack(theta_vec)
            objs = compute_objectives(theta, losses, regularizers, laplacians)
            return objs.total(hyperparameters)

        grad_fn = jax.grad(objective_fn, argnums=0)

        hess_fn = None
        if self.psd_solver.needs_matrix:
            hess_fn = jax.hessian(objective_fn, argnums=0)

        return CompiledNewtonProblem(
            theta_shape=shape,
            losses=losses,
            regularizers=regularizers,
            laplacians=laplacians,
            objective=objective_fn,
            gradient=grad_fn,
            hessian=hess_fn,
        )

    def solve(
        self,
        compiled_problem: CompiledNewtonProblem,
        hyperparameters: Hyperparameters,
    ) -> tuple[StartifiedModel, NewtonSolveInfo]:
        shape = compiled_problem.theta_shape
        dim = int(prod(shape.array_shape))

        def objective_vec(theta_vec: jax.Array) -> jax.Array:
            return compiled_problem.objective(theta_vec, hyperparameters)

        def grad_vec(theta_vec: jax.Array) -> jax.Array:
            return compiled_problem.gradient(theta_vec, hyperparameters)

        hess_vec = None
        hessian_fn = compiled_problem.hessian
        if hessian_fn is not None:

            def hess_vec(theta_vec: jax.Array) -> jax.Array:
                return hessian_fn(theta_vec, hyperparameters)

        initial_theta_vec = jnp.zeros((dim,))
        if compiled_problem.solutions_history:
            candidates = jnp.stack(compiled_problem.solutions_history)
            objectives = jax.vmap(objective_vec)(candidates)
            best_idx = jnp.argmin(objectives)
            initial_theta_vec = candidates[best_idx]

        theta_vec, info = self._solve_vec(
            objective_vec=objective_vec,
            grad_vec=grad_vec,
            hess_vec=hess_vec,
            initial_theta_vec=initial_theta_vec,
        )

        compiled_problem.solutions_history.append(theta_vec)
        if len(compiled_problem.solutions_history) > self.solutions_history_size:
            compiled_problem.solutions_history.pop(0)

        logger.info(
            "newton_solve_complete",
            iterations=info.iterations,
            stopping_reason=info.stopping_reason,
            converged=info.converged(),
        )
        theta = jnp.reshape(theta_vec, shape.array_shape)
        model = StartifiedModel(theta=theta, shape=shape)
        return model, info

    def _solve_vec(
        self,
        *,
        objective_vec: Callable[[jax.Array], jax.Array],
        grad_vec: Callable[[jax.Array], jax.Array],
        hess_vec: Callable[[jax.Array], jax.Array] | None,
        initial_theta_vec: jax.Array,
    ) -> tuple[jax.Array, NewtonSolveInfo]:
        theta_vec = initial_theta_vec
        last_step_alpha: float | None = None
        final_newton_decrement_sq = float("nan")
        previous_step: jax.Array | None = None

        for iteration in range(1, self.max_iters + 1):
            f0 = objective_vec(theta_vec)
            g = grad_vec(theta_vec)
            g_norm = float(jnp.linalg.norm(g))
            logger.debug(
                "newton_iteration",
                iteration=iteration,
                objective_value=float(f0),
                grad_norm=g_norm,
            )

            if g_norm <= self.grad_tol:
                logger.debug(
                    "newton_stop_grad_tol",
                    iteration=iteration,
                    grad_norm=g_norm,
                )
                return theta_vec, NewtonSolveInfo(
                    iterations=iteration,
                    stopping_reason="grad_tol",
                    final_objective_value=float(f0),
                    final_grad_norm=g_norm,
                    final_newton_decrement_sq=final_newton_decrement_sq,
                    last_step_alpha=last_step_alpha,
                )

            step = self._newton_step(
                theta_vec=theta_vec,
                g=g,
                grad_vec=grad_vec,
                hess_vec=hess_vec,
                previous_step=previous_step,
            )
            previous_step = step
            newton_decrement_sq = float(g @ step)
            final_newton_decrement_sq = newton_decrement_sq

            if (0.5 * newton_decrement_sq <= self.newton_atol) or (
                0.5 * newton_decrement_sq <= self.newton_rtol * jnp.abs(f0)
            ):
                logger.debug(
                    "newton_stop_newton_tol",
                    iteration=iteration,
                    newton_decrement_sq=newton_decrement_sq,
                    objective_value=float(f0),
                )
                return theta_vec, NewtonSolveInfo(
                    iterations=iteration,
                    stopping_reason="newton_tol",
                    final_objective_value=float(f0),
                    final_grad_norm=g_norm,
                    final_newton_decrement_sq=final_newton_decrement_sq,
                    last_step_alpha=last_step_alpha,
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
                logger.debug(
                    "newton_backtracking_stop",
                    iteration=iteration,
                    reason=reason,
                    alpha=alpha,
                )
                return theta_vec, NewtonSolveInfo(
                    iterations=iteration,
                    stopping_reason=reason,
                    final_objective_value=float(f0),
                    final_grad_norm=g_norm,
                    final_newton_decrement_sq=final_newton_decrement_sq,
                    last_step_alpha=last_step_alpha,
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
        )

    def _newton_step(
        self,
        *,
        theta_vec: jax.Array,
        g: jax.Array,
        grad_vec: Callable[[jax.Array], jax.Array],
        hess_vec: Callable[[jax.Array], jax.Array] | None,
        previous_step: jax.Array | None,
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
            x0=previous_step,
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
