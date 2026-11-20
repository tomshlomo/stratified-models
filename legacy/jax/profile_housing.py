import cProfile
import pstats

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from stratified_models.pipelines import StratifiedPipelineFitter
from stratified_models.solvers import DirectPSDSolver, Hyperparameters
from stratified_models.solvers.newton import NewtonSolver
from stratified_models.stratifiers import KMeansStratifierFitter, QuantilesBinning


def run_profile() -> None:
    # Setup data (from housing_data fixture)
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    train_df, _test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Features (excluding stratification features)
    exclude = {"target", "HouseAge", "Latitude", "Longitude"}
    feature_cols = [c for c in df.columns if c not in exclude]

    # Setup fitter (from pipeline_fitter fixture)
    house_age_stratifier = QuantilesBinning(out_feature_name="HouseAge_bin", n_bins=5)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
    kmeans_stratifier = KMeansStratifierFitter(
        out_feature_name="Location_cluster", kmeans=kmeans
    )

    solver = NewtonSolver.for_quadratic(DirectPSDSolver())

    pipeline_fitter = StratifiedPipelineFitter(
        stratifiers_fitters=[
            (["HouseAge"], house_age_stratifier),
            (["Latitude", "Longitude"], kmeans_stratifier),
        ],
        regression_features=feature_cols,
        solver=solver,
    )

    # Run fit (from test function)
    compiled = pipeline_fitter.compile(x=train_df.copy(), y=train_df["target"])
    hyperparams = Hyperparameters(
        regularizers={"l2": 0.0},
        graphs={"HouseAge_bin": 0.0, "Location_cluster": 0.0},
    )

    _pipeline, _info = pipeline_fitter.fit(
        compiled=compiled, hyperparameters=hyperparams
    )


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run_profile()
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(30)
