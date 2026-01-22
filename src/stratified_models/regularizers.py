from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

from stratified_models.scalar_function import L1, Array, ScalarFunction, SumOfSquares

L = TypeVar("L", bound=ScalarFunction[Array], covariant=True)


class RegularizationFactory(Generic[L]):
    @abstractmethod
    def build_regularization_function(self, shape: tuple[int, ...]) -> L:
        raise NotImplementedError


class SumOfSquaresRegularizerFactory(RegularizationFactory[SumOfSquares]):
    def build_regularization_function(self, shape: tuple[int, ...]) -> SumOfSquares:
        # todo: we pass the length of each local theta (m). more generally we can pass
        #   the entire shape, and delete the "ReperatedLinearOperator"
        return SumOfSquares(shape=shape[-1])


class L1RegularizationFactory(RegularizationFactory[L1]):
    def build_regularization_function(self, shape: tuple[int, ...]) -> L1:
        return L1()
