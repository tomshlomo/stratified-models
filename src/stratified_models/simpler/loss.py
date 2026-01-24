from abc import ABC, abstractmethod

import attrs

from stratified_models.simpler.scalar_function import (
    Array,
    ScalarFunction,
    SumOfSquaresOverAffine,
)


class Loss(ABC):
    @abstractmethod
    def build(self, x: Array, y: Array) -> ScalarFunction:
        pass


@attrs.frozen(kw_only=True)
class SumOfSquaresLoss(Loss):
    def build(self, x: Array, y: Array) -> SumOfSquaresOverAffine:
        return SumOfSquaresOverAffine(a=x, b=y)
