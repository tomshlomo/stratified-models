from abc import ABC, abstractmethod

import attrs
import jax

from stratified_models.simpler.scalar_function import (
    ScalarFunction,
    SumOfSquaresOverAffine,
)


class Loss(ABC):
    @abstractmethod
    def build(self, x: jax.Array, y: jax.Array) -> ScalarFunction:
        pass


@attrs.frozen(kw_only=True)
class SumOfSquaresLoss(Loss):
    def build(self, x: jax.Array, y: jax.Array) -> SumOfSquaresOverAffine:
        return SumOfSquaresOverAffine(a=x, b=y)
