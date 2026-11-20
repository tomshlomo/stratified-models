from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from stratified_models.loss import SumOfSquaresLoss
from stratified_models.scalar_function import SumOfSquaresOverAffine


def test_sum_of_squares_loss_builds_expected_function() -> None:
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([1.0, 1.0])
    loss = SumOfSquaresLoss()

    func = loss.build(x, y)

    assert isinstance(func, SumOfSquaresOverAffine)
    theta = jnp.array([1.0, 1.0])
    expected = np.sum((np.array([[3.0], [7.0]]) - np.array([[1.0], [1.0]])) ** 2) / 2
    assert np.isclose(float(func(theta)), expected)
