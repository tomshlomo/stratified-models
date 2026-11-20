import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from jsonargparse import CLI
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

from stratified_models.logging_config import configure_logging
from stratified_models.pipelines import StratifiedPipelineFitter
from stratified_models.solvers import DirectPSDSolver
from stratified_models.solvers.newton import NewtonSolver
from stratified_models.stratifiers import KMeansStratifierFitter, QuantilesBinning
from stratified_models.tuning import OptunaTuner

HISTORY_FILE = Path(__file__).parent / "benchmark_history.txt"


def run_benchmark(  # noqa: PLR0913
    kmeans_k: int = 50,
    house_age_bins: int = 10,
    random_seed: int = 42,
    n_folds: int = 3,
    n_trials: int = 20,
    timeout: float | None = None,
    log_level: str = "WARNING",
    # Hyperparameter ranges (log scale)
    l2_range: tuple[float, float] = (1e-6, 1e2),
    graph_range: tuple[float, float] = (1e-4, 1e4),
    # Initial trial multiplier (initial = multiplier * n_rows, None to disable)
    initial_multiplier: float | None = 1e-4,
) -> None:
    configure_logging(
        level=log_level,
        module_levels={"optuna": log_level},
    )

    # Load Data
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_seed)

    # Features
    feature_cols = [
        c
        for c in df.columns
        if c not in ["target", "HouseAge", "Latitude", "Longitude"]
    ]
    all_feature_cols = [c for c in df.columns if c != "target"]

    # Baseline: Linear Regression
    baseline = LinearRegression()
    baseline.fit(train_df[all_feature_cols], train_df["target"])
    baseline_train_r2 = r2_score(
        train_df["target"], baseline.predict(train_df[all_feature_cols])
    )
    baseline_test_r2 = r2_score(
        test_df["target"], baseline.predict(test_df[all_feature_cols])
    )
    print(f"Baseline (Linear Regression) Train R^2: {baseline_train_r2:.4f}")  # noqa: T201
    print(f"Baseline (Linear Regression) Test R^2: {baseline_test_r2:.4f}")  # noqa: T201
    print()  # noqa: T201

    # Stratifiers
    house_age_stratifier = QuantilesBinning(
        out_feature_name="HouseAge_bin", n_bins=house_age_bins
    )
    kmeans = KMeans(
        n_clusters=kmeans_k,
        random_state=random_seed,
        n_init="auto",
    )
    kmeans_stratifier = KMeansStratifierFitter(
        out_feature_name="Location_cluster", kmeans=kmeans
    )

    # Solver
    solver = NewtonSolver.for_quadratic(DirectPSDSolver())

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
        regularizers_range={"l2": l2_range},
        graphs_range={
            "HouseAge_bin": graph_range,
            "Location_cluster": graph_range,
        },
        cv=KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed,
        ),
        n_trials=n_trials,
        timeout=timeout,
        initial_multiplier=initial_multiplier,
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
    stratified_train_r2 = r2_score(train_df["target"], y_train_pred)
    print(f"Stratified Train R^2: {stratified_train_r2:.4f}")  # noqa: T201

    predict_start = time.perf_counter()
    y_test_pred = result.best_pipeline.transform_and_predict(test_df.copy())
    predict_end = time.perf_counter()
    print(f"Predict time: {predict_end - predict_start:.4f}s")  # noqa: T201

    stratified_test_r2 = r2_score(test_df["target"], y_test_pred)
    print(f"Stratified Test R^2: {stratified_test_r2:.4f}")  # noqa: T201

    # Log to history file
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H:%M:%S")
    config = (
        f"k={kmeans_k},bins={house_age_bins},folds={n_folds},trials={n_trials},"
        f"l2_range={l2_range},graph_range={graph_range},init_mult={initial_multiplier}"
    )
    metrics = (
        f"lr_train={baseline_train_r2:.4f},lr_test={baseline_test_r2:.4f},"
        f"strat_train={stratified_train_r2:.4f},strat_test={stratified_test_r2:.4f},"
        f"time={duration:.1f}s"
    )
    with HISTORY_FILE.open("a") as f:
        f.write(f"{timestamp} | {config} | {metrics}\n")
    print(f"\nLogged to {HISTORY_FILE}")  # noqa: T201


if __name__ == "__main__":
    CLI(run_benchmark)
