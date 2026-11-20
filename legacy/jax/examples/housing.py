import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import optuna
    import pandas as pd
    import seaborn as sns
    from sklearn.datasets import fetch_california_housing
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.preprocessing import StandardScaler

    from stratified_models.graph import NetworkXRegularizationGraph
    from stratified_models.loss import SumOfSquaresLoss
    from stratified_models.model import Stratification
    from stratified_models.scalar_function import SumOfSquares
    from stratified_models.solvers.newton import (
        DirectPSDSolver,
        NewtonSolver,
    )
    from stratified_models.problem import (
        AbstractProblem,
        Hyperparameters,
    )
    from stratified_models.stratifiers import (
        ConstantWidth,
        KMeansConfig,
        KMeansStratifier,
    )
    return (
        AbstractProblem,
        ConstantWidth,
        DirectPSDSolver,
        Hyperparameters,
        KFold,
        KMeansConfig,
        KMeansStratifier,
        NewtonSolver,
        Ridge,
        StandardScaler,
        SumOfSquares,
        SumOfSquaresLoss,
        fetch_california_housing,
        mean_squared_error,
        optuna,
        pd,
        r2_score,
        train_test_split,
    )


@app.cell
def _(fetch_california_housing, pd):
    # Load Data
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df
    return (df,)


@app.cell
def _(StandardScaler, df, train_test_split):
    # Preprocessing

    # Split train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Features to regress on (excluding target and stratification feature)
    feature_cols = [c for c in df.columns if c not in ["target", "HouseAge", "Latitude", "Longitude"]]

    # Normalize features
    scaler = StandardScaler()
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    # Add intercept ('ones')
    train_df["ones"] = 1.0
    test_df["ones"] = 1.0

    regression_features = [*feature_cols, "ones"]
    return regression_features, test_df, train_df


@app.cell
def _(ConstantWidth, strat_col, test_df, train_df):
    # Create Stratification and Graph using ConstantWidth

    # We use bin_width=1.0 to mimic the discrete nature of HouseAge (1 to 52)
    stratifier = ConstantWidth(bin_width=1.0).fit(train_df[strat_col])

    # Fit and transform training data
    # We need to pass the column as a DataFrame
    train_strat_df = stratifier.transform(train_df[strat_col]).to_frame()

    # Transform test data
    test_strat_df = stratifier.transform(test_df[strat_col]).to_frame()

    # Update DataFrames with the binned stratification feature
    # We use a new column name to avoid confusion, or overwrite?
    # The stratifier returns a DF with the same column name.
    # But the values are now bin indices (integers).
    # The StratifiedModel will look for 'HouseAge' (strat_col) in X.
    # So we should overwrite 'HouseAge' in the DFs passed to the problem.

    train_df_binned = train_df.copy()
    train_df_binned[strat_col] = train_strat_df[strat_col]

    test_df_binned = test_df.copy()
    test_df_binned[strat_col] = test_strat_df[strat_col]

    reg_graph = stratifier.graph
    return reg_graph, test_df_binned, train_df_binned


@app.cell
def _(
    AbstractProblem,
    SumOfSquares,
    SumOfSquaresLoss,
    reg_graph,
    regression_features,
    train_df_binned,
):
    # Define Problem

    # L2 Regularization on regression coefficients
    # We need to specify the shape of theta: (num_nodes, num_features)
    len(regression_features)

    regularizers = {"l2": SumOfSquares()}

    problem = AbstractProblem(
        x=train_df_binned,
        y=train_df_binned["target"],
        regression_features=regression_features,
        graphs=(reg_graph,),
        loss=SumOfSquaresLoss(),
        regularizers=regularizers,
    )
    return (problem,)


@app.cell
def _(
    AbstractProblem,
    DirectPSDSolver,
    Hyperparameters,
    KFold,
    NewtonSolver,
    SumOfSquares,
    SumOfSquaresLoss,
    mean_squared_error,
    optuna,
    problem,
    reg_graph,
    regression_features,
    train_df_binned,
):
    # Fit Stratified Model with Optuna Tuning

    solver = NewtonSolver.for_quadratic(DirectPSDSolver())

    def objective(trial):
        # Hyperparameters to tune
        graph_reg = trial.suggest_float("graph_reg", 1e-2, 1e4, log=True)
        l2_reg = trial.suggest_float("l2_reg", 1e-4, 1e2, log=True)

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for train_index, val_index in kf.split(train_df_binned):
            fold_train = train_df_binned.iloc[train_index]
            fold_val = train_df_binned.iloc[val_index]

            # Create problem for this fold
            # We reuse the global reg_graph structure (nodes)
            # Nodes with no data in fold_train will have 0 loss term (handled by AbstractProblem)
            fold_problem = AbstractProblem(
                x=fold_train,
                y=fold_train["target"],
                regression_features=regression_features,
                graphs=(reg_graph,),
                loss=SumOfSquaresLoss(),
                regularizers={"l2": SumOfSquares()},
            )

            hyper = Hyperparameters(
                graphs={reg_graph.name: graph_reg},
                regularizers={"l2": l2_reg},
            )

            # Solve
            model_fold, _info, _ = solver.compile_and_solve(fold_problem, hyper)

            # Evaluate on validation fold
            # Filter for known ages to be safe (though graph covers all)
            known_ages = set(model_fold.shape.stratifications[0].index)
            fold_val_filtered = fold_val[fold_val["HouseAge"].isin(known_ages)].copy()

            if len(fold_val_filtered) == 0:
                continue

            y_pred = model_fold.predict(fold_val_filtered)
            y_true = fold_val_filtered["target"]
            mse = mean_squared_error(y_true, y_pred)
            scores.append(mse)

        return sum(scores) / len(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)

    # Retrain on full training set with best params
    best_hyper = Hyperparameters(
        graphs={reg_graph.name: study.best_params["graph_reg"]},
        regularizers={"l2": study.best_params["l2_reg"]},
    )

    model, _info, _ = solver.compile_and_solve(problem, best_hyper)
    return (model,)


@app.cell
def _(mean_squared_error, model, r2_score, test_df_binned):
    # Evaluate Stratified Model

    # We need to handle cases where test set has HouseAge values not in training set
    # The current implementation of predict might fail if index is missing.
    # Let's check if there are unseen ages.
    known_ages = set(model.shape.stratifications[0].index)
    test_df_filtered = test_df_binned[
        test_df_binned["HouseAge"].isin(known_ages)
    ].copy()

    y_pred = model.predict(test_df_filtered)
    y_true = test_df_filtered["target"]

    mse_strat = mean_squared_error(y_true, y_pred)
    r2_strat = r2_score(y_true, y_pred)
    return mse_strat, test_df_filtered


@app.cell
def _(
    Ridge,
    mean_squared_error,
    r2_score,
    regression_features,
    test_df_filtered,
    train_df_binned,
):
    # Baseline: Ridge Regression (Global model)
    X_train = train_df_binned[regression_features]
    y_train = train_df_binned["target"]
    X_test = test_df_filtered[regression_features]
    y_test = test_df_filtered["target"]

    ridge = Ridge(alpha=1.0)  # Matches our l2_reg somewhat, though scaling might differ
    ridge.fit(X_train, y_train)

    y_pred_ridge = ridge.predict(X_test)

    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    return (mse_ridge,)


@app.cell
def _(KMeansConfig, KMeansStratifier, pd, test_df, train_df):
    # Try KMeans Stratification on Latitude/Longitude

    def _run_kmeans_stratification():
        # We will stratify by Location (Lat/Lng)
        loc_cols = ["Latitude", "Longitude"]

        # Try different K values
        k_values = [10, 50, 100]

        results = {}

        for k in k_values:
            print(f"Testing KMeans Stratification with K={k}...")

            stratifier_k = KMeansStratifier.fit(
                pd.Series(
                    map(tuple, train_df[loc_cols].to_numpy()),
                    index=train_df.index,
                    name="LocationCluster",
                ),
                config=KMeansConfig(
                    n_clusters=k,
                    random_state=42,
                    n_init="auto",
                ),
            )

            # Fit on training location data
            # Note: KMeans expects 2D array, which we provide
            train_loc = pd.Series(
                map(tuple, train_df[loc_cols].to_numpy()),
                index=train_df.index,
                name="LocationCluster",
            )

            # Transform
            train_loc_strat = stratifier_k.transform(train_loc)
            test_loc_strat = stratifier_k.transform(
                pd.Series(
                    map(tuple, test_df[loc_cols].to_numpy()),
                    index=test_df.index,
                    name="LocationCluster",
                )
            )

            # Create new DFs with stratification feature
            # We'll call the new feature 'LocationCluster'
            strat_col_k = "LocationCluster"

            train_df_k = train_df.copy()
            train_df_k[strat_col_k] = train_loc_strat.iloc[
                :, 0
            ]  # transform returns DF with one column named after input or default
            # Actually KMeansStratifier names the column based on input.
            # If input is DF with multiple columns, it uses the first column name?
            # Let's check implementation.
            # "if len(X.columns) == 1: self._feature_name = str(X.columns[0]) else: self._feature_name = 'stratification_feature'"
            # So it will be 'stratification_feature' or 'Latitude' (if only Latitude passed).
            # Here we passed 2 columns, so it should be 'stratification_feature'.
            # But wait, transform returns a DF with that name.
            # Let's just assign the values.
            train_df_k[strat_col_k] = train_loc_strat.values.flatten()

            test_df_k = test_df.copy()
            test_df_k[strat_col_k] = test_loc_strat.values.flatten()

            # We need to update the graph's stratification name to match our new column 'LocationCluster'
            # The graph created by stratifier has stratification.name = 'stratification_feature' (likely)
            # We can just rename the column in the DF to match the graph's expected name.
            graph_feature_name = stratifier_k.graph.name
            train_df_k[graph_feature_name] = train_loc_strat.values.flatten()
            test_df_k[graph_feature_name] = test_loc_strat.values.flatten()

            results[k] = (stratifier_k.graph, train_df_k, test_df_k, graph_feature_name)

        return k_values, loc_cols, results

    k_values, loc_cols, results = _run_kmeans_stratification()
    return (results,)


@app.cell
def _(
    AbstractProblem,
    DirectPSDSolver,
    Hyperparameters,
    KFold,
    NewtonSolver,
    SumOfSquares,
    SumOfSquaresLoss,
    mean_squared_error,
    optuna,
    regression_features,
    results,
):
    # Evaluate KMeans Stratified Models

    def _evaluate_kmeans():
        solver_kmeans = NewtonSolver.for_quadratic(DirectPSDSolver())

        best_scores_kmeans = {}

        for k, (reg_graph_k, train_df_k, test_df_k, strat_col_k) in results.items():
            print(f"\nOptimizing for K={k}...")

            def objective_kmeans(trial):
                graph_reg = trial.suggest_float("graph_reg", 1e-2, 1e4, log=True)
                l2_reg = trial.suggest_float("l2_reg", 1e-4, 1e2, log=True)

                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                scores = []

                for train_index, val_index in kf.split(train_df_k):
                    fold_train = train_df_k.iloc[train_index]
                    fold_val = train_df_k.iloc[val_index]

                    fold_problem = AbstractProblem(
                        x=fold_train,
                        y=fold_train["target"],
                        regression_features=regression_features,
                        graphs=(reg_graph_k,),
                        loss=SumOfSquaresLoss(),
                        regularizers={"l2": SumOfSquares()},
                    )

                    hyper = Hyperparameters(
                        graphs={reg_graph_k.name: graph_reg},
                        regularizers={"l2": l2_reg},
                    )

                    model_fold, _info, _ = solver_kmeans.compile_and_solve(
                        fold_problem, hyper
                    )

                    # Filter for known clusters (KMeans should cover all, but just in case of empty clusters in fold)
                    known_clusters = set(model_fold.shape.stratifications[0].index)
                    fold_val_filtered = fold_val[
                        fold_val[strat_col_k].isin(known_clusters)
                    ].copy()

                    if len(fold_val_filtered) == 0:
                        continue

                    y_pred = model_fold.predict(fold_val_filtered)
                    y_true = fold_val_filtered["target"]
                    mse = mean_squared_error(y_true, y_pred)
                    scores.append(mse)

                return sum(scores) / len(scores)

        study_kmeans = optuna.create_study(direction="minimize")
        study_kmeans.optimize(objective_kmeans, n_trials=2)  # Fewer trials for speed

        print(f"Best params for K={k}: {study_kmeans.best_params}")

        # Retrain on full training set
        best_hyper_kmeans = Hyperparameters(
            graphs={reg_graph_k.name: study_kmeans.best_params["graph_reg"]},
            regularizers={"l2": study_kmeans.best_params["l2_reg"]},
        )

        problem_kmeans = AbstractProblem(
            x=train_df_k,
            y=train_df_k["target"],
            regression_features=regression_features,
            graphs=(reg_graph_k,),
            loss=SumOfSquaresLoss(),
            regularizers={"l2": SumOfSquares()},
        )

        model_kmeans, _info, _ = solver_kmeans.compile_and_solve(
            problem_kmeans, best_hyper_kmeans
        )

        # Evaluate on test set
        known_clusters = set(model_kmeans.shape.stratifications[0].index)
        test_df_filtered_kmeans = test_df_k[
            test_df_k[strat_col_k].isin(known_clusters)
        ].copy()

        y_pred_kmeans = model_kmeans.predict(test_df_filtered_kmeans)
        y_true_kmeans = test_df_filtered_kmeans["target"]
        mse_kmeans = mean_squared_error(y_true_kmeans, y_pred_kmeans)

        print(f"Test MSE for K={k}: {mse_kmeans:.4f}")
        best_scores_kmeans[k] = mse_kmeans

        return best_scores_kmeans, solver_kmeans

    best_scores_kmeans, solver_kmeans = _evaluate_kmeans()
    return (best_scores_kmeans,)


@app.cell
def _(best_scores_kmeans, mse_ridge, mse_strat):
    print("\nComparison:")
    print(f"Ridge MSE: {mse_ridge:.4f}")
    print(f"HouseAge Stratified MSE: {mse_strat:.4f}")
    for k, score in best_scores_kmeans.items():
        print(f"Location Stratified (K={k}) MSE: {score:.4f}")
    return


if __name__ == "__main__":
    app.run()
