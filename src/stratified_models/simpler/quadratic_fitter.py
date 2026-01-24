from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import attrs
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from stratified_models.simpler.fit import AbstractProblem, Hyperparameters, Solver
from stratified_models.simpler.linear_operator import LinearOperator
from stratified_models.simpler.model import StartifiedModel, ThetaShape
from stratified_models.simpler.quadratic import ExplicitQuadraticFunction
from stratified_models.simpler.scalar_function import Array, QuadraticScalarFunction


@attrs.frozen(kw_only=True)
class CompiledQuadraticProblem:
    theta_shape: ThetaShape
    loss: ExplicitQuadraticFunction
    regularizers: Mapping[str, ExplicitQuadraticFunction]
    laplacians: Mapping[str, ExplicitQuadraticFunction]


class PSDSystemSolver(ABC):
    @abstractmethod
    def solve(self, f: ExplicitQuadraticFunction) -> Array:
        """Solve argmin_x xᵀQx/2 + cᵀx + d."""


class DirectSolver(PSDSystemSolver):
    def solve(self, f: ExplicitQuadraticFunction) -> Array:
        x = scipy.sparse.linalg.spsolve(f.q.as_sparse_matrix(), -f.c)
        return np.asarray(x, dtype=float)


@attrs.frozen(kw_only=True)
class CGSolver(PSDSystemSolver):
    atol: float = 1e-8
    rtol: float = 1e-8
    max_iter: int | None = None

    def solve(self, f: ExplicitQuadraticFunction) -> Array:
        x, info = scipy.sparse.linalg.cg(
            f.q.to_scipy_linear_operator(),
            -f.c,
            atol=self.atol,
            rtol=self.rtol,
            maxiter=self.max_iter,
        )
        if info != 0:
            raise RuntimeError
        return np.asarray(x, dtype=float)


@attrs.frozen(kw_only=True)
class QuadraticSolver(Solver[CompiledQuadraticProblem]):
    solver: PSDSystemSolver

    def compile(self, abstract_problem: AbstractProblem) -> CompiledQuadraticProblem:
        theta_shape = abstract_problem.theta_shape
        loss = _build_loss_quadratic(problem=abstract_problem)

        regularizers: dict[str, ExplicitQuadraticFunction] = {}
        for name, func in abstract_problem.regularizers.items():
            if not isinstance(func, QuadraticScalarFunction):
                raise TypeError
            regularizers[name] = func.to_explicit_quadratic()

        laplacians: dict[str, ExplicitQuadraticFunction] = {}
        c_to_f, f_to_c = _c_f_permutations(
            num_nodes=theta_shape.num_nodes,
            m=theta_shape.m,
        )
        for name, func in abstract_problem.laplacians():
            if not isinstance(func, QuadraticScalarFunction):
                raise TypeError
            lap_f = func.to_explicit_quadratic()
            laplacians[name] = ExplicitQuadraticFunction(
                q=_COrderLinearOperatorFromFOrder(
                    base=lap_f.q,
                    c_to_f=c_to_f,
                    f_to_c=f_to_c,
                ),
                c=np.asarray(lap_f.c, dtype=float)[f_to_c],
                d=lap_f.d,
            )

        return CompiledQuadraticProblem(
            theta_shape=theta_shape,
            loss=loss,
            regularizers=regularizers,
            laplacians=laplacians,
        )

    def solve(
        self,
        compiled_problem: CompiledQuadraticProblem,
        hyperparameters: Hyperparameters,
    ) -> StartifiedModel:
        size = compiled_problem.theta_shape.num_nodes * compiled_problem.theta_shape.m
        components: list[tuple[ExplicitQuadraticFunction, float]] = [
            (compiled_problem.loss, 1.0),
        ]
        for name, gamma in hyperparameters.regularizers.items():
            components.append((compiled_problem.regularizers[name], gamma))
        for name, gamma in hyperparameters.graphs.items():
            components.append((compiled_problem.laplacians[name], gamma))

        f = ExplicitQuadraticFunction.sum(
            m=size,
            components=tuple(components),
        )
        theta_vec = self.solver.solve(f)
        num_nodes = compiled_problem.theta_shape.num_nodes
        m = compiled_problem.theta_shape.m
        theta_matrix = theta_vec.reshape((num_nodes, m), order="C")
        return StartifiedModel.from_array(
            arr=theta_matrix,
            shape=compiled_problem.theta_shape,
        )


def _build_loss_quadratic(*, problem: AbstractProblem) -> ExplicitQuadraticFunction:
    """Build loss quadratic in row-major (node-major) coordinates.

    The optimization variable is interpreted as a `(num_nodes, m)` matrix and
    vectorized as `theta.ravel(order="C")`, i.e. consecutive blocks per node.
    """
    theta_shape = problem.theta_shape
    num_nodes = theta_shape.num_nodes
    m = theta_shape.m

    components: dict[int, ExplicitQuadraticFunction] = {}

    for node, loss in problem.group_losses():
        if not isinstance(loss, QuadraticScalarFunction):
            raise TypeError

        node_index = theta_shape.node_to_flat_index(node)
        components[node_index] = loss.to_explicit_quadratic()

    return ExplicitQuadraticFunction.concat(
        k=num_nodes,
        m=m,
        components=components,
    )


def _c_f_permutations(*, num_nodes: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Return permutations between C-order and F-order vectorizations.

    Let `theta` be shaped `(num_nodes, m)`.

    - C-order: `x_c = theta.ravel(order="C")` (node-major)
    - F-order: `x_f = theta.ravel(order="F")` (feature-major)
    """
    n = num_nodes * m
    c_to_f = (
        np.arange(n, dtype=int).reshape((num_nodes, m), order="C").T.ravel(order="C")
    )
    f_to_c = np.empty(n, dtype=int)
    f_to_c[c_to_f] = np.arange(n, dtype=int)
    return c_to_f, f_to_c


@attrs.frozen(kw_only=True)
class _COrderLinearOperatorFromFOrder(LinearOperator):
    """Wrap an operator defined on vec_F(theta) to act on vec_C(theta)."""

    base: LinearOperator
    c_to_f: np.ndarray
    f_to_c: np.ndarray

    def size(self) -> int:
        return self.base.size()

    def matvec(self, x: Array) -> Array:
        x_f = x[self.c_to_f]
        y_f = self.base.matvec(x_f)
        return y_f[self.f_to_c]

    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        n = self.size()
        p = scipy.sparse.csr_matrix(
            (np.ones(n), (np.arange(n), self.c_to_f)),
            shape=(n, n),
        )
        q_f = self.base.as_sparse_matrix()
        return p.T @ q_f @ p
