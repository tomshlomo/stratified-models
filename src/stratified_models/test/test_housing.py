"""Tests using the California Housing dataset."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from stratified_models.pipelines import StratifiedPipelineFitter
from stratified_models.solvers import DirectPSDSolver, Hyperparameters
from stratified_models.solvers.newton import NewtonSolver
from stratified_models.stratifiers import KMeansStratifierFitter, QuantilesBinning


@pytest.fixture
def housing_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load California housing data and return train/test splits."""
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Features (excluding stratification features)
    exclude = {"target", "HouseAge", "Latitude", "Longitude"}
    feature_cols = [c for c in df.columns if c not in exclude]

    return train_df, test_df, feature_cols


@pytest.fixture
def pipeline_fitter(
    housing_data: tuple[pd.DataFrame, pd.DataFrame, list[str]],
) -> StratifiedPipelineFitter:
    """Create a pipeline fitter for housing data."""
    _, _, feature_cols = housing_data

    house_age_stratifier = QuantilesBinning(out_feature_name="HouseAge_bin", n_bins=5)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
    kmeans_stratifier = KMeansStratifierFitter(
        out_feature_name="Location_cluster", kmeans=kmeans
    )

    solver = NewtonSolver.for_quadratic(DirectPSDSolver())
    return StratifiedPipelineFitter(
        stratifiers_fitters=[
            (["HouseAge"], house_age_stratifier),
            (["Latitude", "Longitude"], kmeans_stratifier),
        ],
        regression_features=feature_cols,
        solver=solver,
    )


def test_zero_regularization_beats_linear_regression_on_train(
    housing_data: tuple[pd.DataFrame, pd.DataFrame, list[str]],
    pipeline_fitter: StratifiedPipelineFitter,
) -> None:
    """Stratified model with no regularization should have higher train R² than LR."""
    train_df, _, _ = housing_data
    all_features = [c for c in train_df.columns if c != "target"]

    # Baseline: Linear Regression
    lr = LinearRegression()
    lr.fit(train_df[all_features], train_df["target"])
    lr_train_r2 = r2_score(train_df["target"], lr.predict(train_df[all_features]))

    # Stratified model with zero regularization
    compiled = pipeline_fitter.compile(x=train_df.copy(), y=train_df["target"])
    hyperparams = Hyperparameters(
        regularizers={"l2": 0.0},
        graphs={"HouseAge_bin": 0.0, "Location_cluster": 0.0},
    )
    pipeline, info = pipeline_fitter.fit(compiled=compiled, hyperparameters=hyperparams)

    assert info.converged(), f"Solver did not converge: {info}"

    train_pred = pipeline.transform_and_predict(train_df.copy())
    stratified_train_r2 = r2_score(train_df["target"], train_pred)

    # Stratified model has more parameters, so should fit training data better
    assert stratified_train_r2 > lr_train_r2, (
        f"Expected stratified train R² ({stratified_train_r2:.4f}) > "
        f"LR train R² ({lr_train_r2:.4f})"
    )
