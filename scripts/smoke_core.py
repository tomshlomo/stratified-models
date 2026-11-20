"""Check a clean core-wheel installation without optional integrations."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np

from stratified_models import (
    L2,
    AddIntercept,
    BinaryLogisticSpec,
    FitFailure,
    NewtonSolver,
    Pipeline,
    QuadraticSolver,
    RegressionSpec,
    compose,
    fit,
)


def main() -> None:
    optional = ("pandas", "polars", "pyarrow", "sklearn", "optuna", "cvxpy", "marimo")
    installed = [
        name for name in optional if importlib.util.find_spec(name) is not None
    ]
    if installed:
        raise RuntimeError(
            f"This smoke check requires a core-only environment: {installed}"
        )
    x = np.arange(12, dtype=float)
    recipe = Pipeline(
        preprocessing=compose(AddIntercept(name="intercept")),
        model=RegressionSpec(features=("x", "intercept")),
    )
    result = fit(recipe, {"x": x}, 2 * x + 3, solver=QuadraticSolver())
    if isinstance(result, FitFailure):
        raise RuntimeError(result.reason.message)
    np.testing.assert_allclose(result.predictor.predict({"x": x}), 2 * x + 3)
    classification = Pipeline(
        preprocessing=compose(AddIntercept(name="intercept")),
        model=BinaryLogisticSpec(
            features=("x", "intercept"),
            penalties=(L2(name="ridge", weight=0.1, features=("x",)),),
        ),
    )
    binary = fit(
        classification, {"x": x - 5.5}, (x > 5).astype(float), solver=NewtonSolver()
    )
    if isinstance(binary, FitFailure):
        raise RuntimeError(binary.reason.message)
    probabilities = binary.predictor.predict_proba({"x": x - 5.5})
    assert probabilities.shape == (12, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    imported = [name for name in optional if name in sys.modules]
    if imported:
        raise RuntimeError(f"Core fitting imported optional integrations: {imported}")
    print(
        "Core wheel fits regression and logistic models without optional integrations."
    )


if __name__ == "__main__":
    main()
