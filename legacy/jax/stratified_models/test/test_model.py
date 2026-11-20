from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd

from stratified_models.model import StartifiedModel, Stratification, ThetaShape


def test_theta_shape_properties_and_node_to_index() -> None:
    strat_a = Stratification(index=pd.Index(["low", "high"], name="a"))
    strat_b = Stratification(index=pd.Index([10, 20, 30], name="b"))
    shape = ThetaShape(
        regression_features=["x1", "x2"],
        stratifications=[strat_a, strat_b],
    )

    assert shape.m == 2
    assert shape.stratification_features == ("a", "b")
    assert shape.graph_sizes == (2, 3)
    assert shape.array_shape == (2, 3, 2)
    assert shape.num_nodes == 6
    assert shape.node_to_index(("high", 20)) == (1, 1)


def test_model_theta_at_node_and_predict() -> None:
    strat_a = Stratification(index=pd.Index(["low", "high"], name="a"))
    strat_b = Stratification(index=pd.Index([10, 20], name="b"))
    shape = ThetaShape(
        regression_features=["x1", "x2"],
        stratifications=[strat_a, strat_b],
    )
    theta = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )
    model = StartifiedModel(theta=theta, shape=shape)

    assert np.allclose(
        np.asarray(model.theta_at_node(("high", 20))), np.array([0.0, 2.0])
    )

    x = pd.DataFrame(
        {
            "a": ["low", "high", "low"],
            "b": [10, 20, 20],
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
        },
        index=[5, 6, 7],
    )
    predicted = model.predict(x)

    expected = np.array(
        [
            1.0 * 1.0 + 4.0 * 0.0,
            2.0 * 0.0 + 5.0 * 2.0,
            3.0 * 0.0 + 6.0 * 1.0,
        ]
    )
    assert np.allclose(predicted.to_numpy(), expected)
    assert list(predicted.index) == [5, 6, 7]
