from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stratified_models.linear_operator import (
    Array,
    BlockDiagonalLinearOperator,
    LinearOperator,
    RepeatedLinearOperator,
    SumOfLinearOperators,
)

# class QuadraticFunction(Protocol):
#     def __call__(self, x: Vector) -> float:
#         pass
#
#     def to_explicit_quadratic(self) -> ExplicitQuadraticFunction:
#         pass
# def get_q(self) -> LinearOperator:
#     pass
#
# def get_c(self) -> Vector:
#     pass
#
# def get_d(self) -> float:
#     pass


@dataclass
class ExplicitQuadraticFunction:
    q: LinearOperator
    c: Array
    d: float

    def __call__(self, x: Array) -> float:
        return float((x @ (self.q.matvec(x))) / 2 + x @ self.c + self.d / 2)

    @classmethod
    def quadratic_form(cls, q: LinearOperator) -> ExplicitQuadraticFunction:
        return ExplicitQuadraticFunction(
            q=q,
            c=np.zeros(q.size()),
            d=0.0,
        )

    @classmethod
    def sum(
        cls, m: int, components: list[tuple[ExplicitQuadraticFunction, float]]
    ) -> ExplicitQuadraticFunction:
        # todo: if empty return the zero quadratic
        q = []
        c = np.zeros(m)
        d = 0.0
        for f, gamma in components:
            q.append((f.q, gamma))
            c += f.c * gamma
            d += f.d * gamma
        return ExplicitQuadraticFunction(
            q=SumOfLinearOperators(q, m),
            c=c,
            d=d,
        )

    @classmethod
    def concat(
        cls, k: int, m: int, components: dict[int, ExplicitQuadraticFunction]
    ) -> ExplicitQuadraticFunction:
        q = {}
        c = np.zeros((k, m))
        d = 0.0
        for i, f in components.items():
            q[i] = f.q
            c[i] = f.c
            d += f.d
        return ExplicitQuadraticFunction(
            q=BlockDiagonalLinearOperator(blocks=q, k=k, m=m),
            c=c.ravel(),
            d=d,
        )

    def repeat(self, repetitions: int) -> ExplicitQuadraticFunction:
        return ExplicitQuadraticFunction(
            q=RepeatedLinearOperator(self.q, repetitions),
            c=self.c.repeat(repetitions),
            d=self.d * repetitions,
        )


#
# class BlockDiagonalLinearOperator2(LinearOperator):
#     def __init__(
#             self,
#             q: NumpyArray,
#     ) -> None:
#         self.k, self.m, _ = q.shape
#         self.q = q
#         self.dtype = self.q.dtype
#         # mk = self.m * self.k
#         # self.shape = (mk, mk)
#         # self.dtype = q.dtype
#
#     def matvec(self, x: Vector) -> Vector:
#         # x = x.reshape((self.k, self.m))
#         return np.einsum("kim,ki->km", self.q, x)
#         # return qx.ravel()  # type:ignore[no-any-return]
#
#     def ravel_matvec(self, x: NumpyArray) -> NumpyArray:
#         x = x.reshape((self.k, self.m))
#         return self.matvec(x).ravel()
#
#     def as_matrix(self) -> scipy.sparse.spmatrix:
#         return scipy.sparse.block_diag(self.q)
#
#     def as_scipy_linear_operator(self) -> CallbackBasedScipyLinearOperator:
#         return CallbackBasedScipyLinearOperator(
#             matvec=partial(self.ravel_matvec, self),
#             shape=(self.k * self.m, self.k * self.m),
#             dtype=self.dtype,
#         )
#
#     def quad_form_prox(self, v, rho):
#         """
#         argmin x' a x + rho/2 |x-v|^2
#         2 a x + rho (x-v) = 0
#         (2a + rho I) x = rho v
#         x = (2/rho a + I)^-1 v
#         :param v:
#         :param rho:
#         :return:
#         """
#         p = 2 / rho * self.q + np.eye(self.m)
#         x = np.zeros((self.k, self.m))
#         for i in range(self.k):
#             x[i] = np.linalg.solve(p[i], v[i])
#         return x
#
#     def traces(self):
#         # todo: rewrite using einsum
#         x = np.zeros(self.k)
#         for i in range(self.k):
#             x[i] = np.trace(self.q[i])
#         return x
#
#
# class CallbackBasedScipyLinearOperator(scipy.sparse.linalg.LinearOperator):
#     def __init__(
#             self,
#             matvec: Callable[[Vector], Vector],
#             shape: tuple[int, int],
#             dtype: type,
#     ):
#         self._matvec = matvec
#         self.shape = shape
#         self.dtype = dtype
