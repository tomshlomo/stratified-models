from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from stratified_models.graph import NetworkXRegularizationGraph
from stratified_models.model import StartifiedModel, Stratification, ThetaShape
from stratified_models.pipelines import StratifiedPipeline, StratifiedPipelineFitter
from stratified_models.solvers import (
    AbstractProblem,
    Hyperparameters,
    SolveInfo,
    Solver,
)
from stratified_models.stratifiers import BinningStratifier, ConstantWidthBinning


class DummySolveInfo(SolveInfo):
    def converged(self) -> bool:
        return True


class DummySolver(Solver[AbstractProblem, DummySolveInfo]):
    def __init__(self) -> None:
        self.compiled: AbstractProblem | None = None
        self.last_hyperparameters: Hyperparameters | None = None

    def compile(self, abstract_problem: AbstractProblem) -> AbstractProblem:
        self.compiled = abstract_problem
        return abstract_problem

    def solve(
        self,
        compiled_problem: AbstractProblem,
        hyperparameters: Hyperparameters,
    ) -> tuple[StartifiedModel, DummySolveInfo]:
        self.last_hyperparameters = hyperparameters
        theta_shape = compiled_problem.theta_shape
        theta = jnp.zeros(theta_shape.array_shape)
        return StartifiedModel(theta=theta, shape=theta_shape), DummySolveInfo()


def test_pipeline_predict_applies_stratifier_and_normalizer() -> None:
    stratification = Stratification(index=pd.Index([0, 1], name="group"))
    shape = ThetaShape(
        regression_features=["x1", "x2"], stratifications=[stratification]
    )
    theta = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    model = StartifiedModel(theta=theta, shape=shape)
    graph = NetworkXRegularizationGraph.path(2, name="group")
    stratifier = BinningStratifier(
        bin_edges=np.array([0.5]), graph=graph, in_feature_name="value"
    )
    x = pd.DataFrame({"value": [0.1, 0.6], "x1": [1.0, 2.0], "x2": [3.0, 4.0]})
    normalizer = StandardScaler().fit(x[["x1", "x2"]])
    pipeline = StratifiedPipeline(
        stratifiers=[stratifier], normalizer=normalizer, model=model
    )

    x_work = x.copy()
    predicted = pipeline.predict(x_work)

    normalized = normalizer.transform(x[["x1", "x2"]])
    theta_aligned = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.sum(normalized * theta_aligned, axis=1)
    assert np.allclose(predicted.to_numpy(), expected)
    assert "group" in x_work.columns


def test_pipeline_fitter_builds_problem_and_mutates_frame() -> None:
    x = pd.DataFrame(
        {"value": [0.0, 1.0, 2.0], "x1": [1.0, 2.0, 3.0], "x2": [2.0, 1.0, 0.0]}
    )
    y = pd.Series([1.0, 2.0, 3.0], index=x.index)
    stratifier_fitter = ConstantWidthBinning(
        out_feature_name="group",
        min_width=1.0,
        min_num_bins=2,
    )
    solver = DummySolver()
    fitter = StratifiedPipelineFitter(
        stratifiers_fitters=[(["value"], stratifier_fitter)],
        regression_features=["x1", "x2"],
        solver=solver,
        normalize=True,
        intercept=True,
    )
    hyperparameters = Hyperparameters(graphs={"group": 1.0}, regularizers={})

    pipeline, info = fitter.fit(x, y, hyperparameters)

    assert solver.compiled is not None
    assert solver.last_hyperparameters is hyperparameters
    assert "group" in x.columns
    assert "one" in x.columns
    assert np.allclose(x[["x1", "x2"]].mean().to_numpy(), 0.0, atol=1e-7)
    assert len(solver.compiled.graphs) == 1
    assert pipeline.normalizer is not None
    assert len(pipeline.stratifiers) == 1
    assert info.converged()
