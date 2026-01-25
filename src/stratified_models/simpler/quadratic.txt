from __future__ import annotations

import attrs
import numpy as np

from stratified_models.simpler.linear_operator import (
    Array,
    BlockDiagonalLinearOperator,
    LinearOperator,
    RepeatedLinearOperator,
    SumOfLinearOperators,
)


@attrs.frozen(kw_only=True)
class ExplicitQuadraticFunction:
    """x * (q * x) / 2 + (c * x) + d."""

    q: LinearOperator
    c: Array
    d: float

    def __call__(self, x: Array) -> float:
        return float((x @ (self.q.matvec(x))) / 2 + x @ self.c + self.d)

    @staticmethod
    def quadratic_form(q: LinearOperator) -> ExplicitQuadraticFunction:
        return ExplicitQuadraticFunction(
            q=q,
            c=np.zeros(q.size()),
            d=0.0,
        )

    @staticmethod
    def sum(
        m: int,
        components: tuple[tuple[ExplicitQuadraticFunction, float], ...],
    ) -> ExplicitQuadraticFunction:
        # TODO: if empty return the zero quadratic
        q = []
        c = np.zeros(m)
        d = 0.0
        for f, gamma in components:
            q.append((f.q, gamma))
            c += f.c * gamma
            d += f.d * gamma
        return ExplicitQuadraticFunction(
            q=SumOfLinearOperators(tuple(q), m),
            c=c,
            d=d,
        )

    @classmethod
    def concat(
        cls,
        k: int,
        m: int,
        components: dict[int, ExplicitQuadraticFunction],
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
