import time

import attrs
import pandas as pd
from jsonargparse import CLI
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

from stratified_models.pipelines import StratifiedPipelineFitter
from stratified_models.solvers.newton import CGPSDSolver, NewtonSolver
from stratified_models.stratifiers import KMeansStratifierFitter, QuantilesBinning
from stratified_models.tuning import OptunaTuner


@attrs.frozen(kw_only=True)
class BenchmarkConfig:
    kmeans_k: int = 50
    house_age_bins: int = 10
    newton_max_iters: int = 10
    cg_max_iters: int = 100
    random_seed: int = 42
    n_folds: int = 3
    n_trials: int = 20
    timeout: float | None = None


def run_benchmark(benchmark_config: BenchmarkConfig) -> None:
    print(f"Running benchmark with config: {benchmark_config}")  # noqa: T201

    # Load Data
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    # Split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=benchmark_config.random_seed
    )

    # Features
    feature_cols = [
        c
        for c in df.columns
        if c not in ["target", "HouseAge", "Latitude", "Longitude"]
    ]

    # Stratifiers
    house_age_stratifier = QuantilesBinning(
        out_feature_name="HouseAge_bin", n_bins=benchmark_config.house_age_bins
    )
    kmeans = KMeans(
        n_clusters=benchmark_config.kmeans_k,
        random_state=benchmark_config.random_seed,
        n_init="auto",
    )
    kmeans_stratifier = KMeansStratifierFitter(
        out_feature_name="Location_cluster", kmeans=kmeans
    )

    # Solver
    psd_solver = CGPSDSolver(maxiter=benchmark_config.cg_max_iters)
    solver = NewtonSolver(
        max_iters=benchmark_config.newton_max_iters, psd_solver=psd_solver
    )

    # Pipeline fitter
    pipeline_fitter = StratifiedPipelineFitter(
        stratifiers_fitters=[
            (["HouseAge"], house_age_stratifier),
            (["Latitude", "Longitude"], kmeans_stratifier),
        ],
        regression_features=feature_cols,
        solver=solver,
    )

    # Tuner
    tuner = OptunaTuner(
        pipeline_fitter=pipeline_fitter,
        regularizers_range={"l2": (1e-6, 1e2)},
        graphs_range={
            "HouseAge_bin": (1e-4, 1e4),
            "Location_cluster": (1e-4, 1e4),
        },
        cv=KFold(
            n_splits=benchmark_config.n_folds,
            shuffle=True,
            random_state=benchmark_config.random_seed,
        ),
        n_trials=benchmark_config.n_trials,
        timeout=benchmark_config.timeout,
    )

    print("Starting tuning...")  # noqa: T201
    start_time = time.perf_counter()

    result = tuner.tune(x=train_df.copy(), y=train_df["target"])

    end_time = time.perf_counter()
    duration = end_time - start_time

    print(f"\nTuning complete in {duration:.4f} seconds.")  # noqa: T201
    print(f"Best hyperparameters: {result.best_hyperparameters}")  # noqa: T201
    print(f"Best CV R^2: {result.study.best_value:.4f}")  # noqa: T201
    print(f"Solver info: {result.best_solve_info}")  # noqa: T201

    # Evaluate
    y_train_pred = result.best_pipeline.transform_and_predict(train_df.copy())
    train_r2 = r2_score(train_df["target"], y_train_pred)
    print(f"Train R^2: {train_r2:.4f}")  # noqa: T201

    predict_start = time.perf_counter()
    y_test_pred = result.best_pipeline.transform_and_predict(test_df.copy())
    predict_end = time.perf_counter()
    print(f"Predict time: {predict_end - predict_start:.4f}s")  # noqa: T201

    test_r2 = r2_score(test_df["target"], y_test_pred)
    print(f"Test R^2: {test_r2:.4f}")  # noqa: T201


if __name__ == "__main__":
    CLI(run_benchmark)
