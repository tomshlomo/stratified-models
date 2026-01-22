from __future__ import annotations

from itertools import product
from typing import Any, Callable, Hashable, Optional, Self

import networkx as nx
import numpy as np
import optuna
import pandas as pd
import scipy
from numpy.typing import NDArray
from optuna.visualization import plot_slice
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, StandardScaler
from tqdm import tqdm

from stratified_models.estimator import StratifiedLogisticRegressionClassifier
from stratified_models.fitters.admm_fitter import ADMMFitter
from stratified_models.fitters.cvxpy_fitter import CVXPYFitter
from stratified_models.regularization_graph.networkx_graph import (
    NetworkXRegularizationGraph,
)
from stratified_models.regularizers import SumOfSquaresRegularizerFactory

Array = NDArray[np.float64]
set_config(transform_output="pandas")

df = pd.read_csv("examples/data/heart.csv")
df.describe()
df = pd.get_dummies(df, drop_first=True)
df["one"] = 1.0
y = df["HeartDisease"]
X = df.drop(columns=["HeartDisease"])
# numerical_features = X.drop(['HeartDisease'], axis=1).select_dtypes('number').columns
# categorical_features = X.select_dtypes('object').columns

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
print(f"{x_train.shape=}, {x_test.shape=}")


def evaluate_model(model: BaseEstimator) -> None:
    model.fit(x_train, y_train)
    y_train_prob = model.predict_proba(x_train)[:, 1]
    y_train_bin = model.predict(x_train)

    acc_train = accuracy_score(y_train, y_train_bin)
    roc_train = roc_auc_score(y_train, y_train_prob)
    ce_train = log_loss(y_train, y_train_prob)

    y_test_prob = model.predict_proba(x_test)[:, 1]
    y_test_bin = model.predict(x_test)

    acc_test = accuracy_score(y_test, y_test_bin)
    roc_test = roc_auc_score(y_test, y_test_prob)
    ce_test = log_loss(y_test, y_test_prob)

    print(f"{acc_train=:.3f}, {acc_test=:.3f}")
    print(f"{roc_train=:.3f}, {roc_test=:.3f}")
    print(f"{ce_train=:.3f}, {ce_test=:.3f}")
    return acc_test, roc_test, ce_test


def main():
    features = [
        "one",
        "Sex_M",
        "Age",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "Oldpeak",
        "ChestPainType_ATA",
        "ChestPainType_NAP",
        "ChestPainType_TA",
        "RestingECG_Normal",
        "RestingECG_ST",
        "ExerciseAngina_Y",
        "ST_Slope_Flat",
        "ST_Slope_Up",
    ]
    simple_linear = Pipeline(
        [
            (
                "transform",
                ColumnTransformer(
                    [
                        ("selector", StandardScaler(), features[1:]),
                        ("none", "passthrough", ["one"]),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            ("linear", LogisticRegression(penalty=None, fit_intercept=False)),
        ]
    )
    evaluate_model(simple_linear)

    simple_linear = Pipeline(
        [
            (
                "transform",
                ColumnTransformer(
                    [
                        ("selector", StandardScaler(), features[1:]),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            ("linear", LogisticRegression(penalty=None, fit_intercept=True)),
        ]
    )
    evaluate_model(simple_linear)

    def objective(trial) -> float:
        # alpha = 1e-2 * 300
        # alpha = 0.0
        # beta = 1e2 * 300
        # beta = 0.0
        # n_bins = 2
        alpha = trial.suggest_float("alpha", 1e-9, 1e3, log=True)
        beta = trial.suggest_float("beta", 1e-9, 1e3, log=True)
        gamma = trial.suggest_float("gamma", 1e-9, 1e3, log=True)
        # n_bins = trial.suggest_int('nbins', 5, 30)
        # age_graph = nx.path_graph(n_bins)
        sex_graph = nx.path_graph(2)
        # graph = nx.path_graph(1)
        model = Pipeline(
            [
                (
                    "transform",
                    ColumnTransformer(
                        [
                            ("none", "passthrough", ["one", "Sex_M"]),
                            # ('scale', StandardScaler(), regression_features[2:]),
                            ("scale", StandardScaler(), features[2:]),
                            # (
                            #     'discretize',
                            #     KBinsDiscretizer(n_bins=n_bins, strategy='uniform', encode='ordinal'),
                            #     ['Age'],
                            # ),
                        ],
                        verbose_feature_names_out=False,
                    ),
                ),
                (
                    "to_int",
                    ColumnTransformer(
                        [
                            (
                                "to_int",
                                FunctionTransformer(lambda x: np.isnan(x).astype(int)),
                                ["Age", "Sex_M"],
                            )
                        ],
                        verbose_feature_names_out=False,
                        remainder="passthrough",
                    ),
                ),
                (
                    "predict",
                    StratifiedLogisticRegressionClassifier(
                        # regularizers_factories=tuple(),
                        regularizers_factories=(
                            (SumOfSquaresRegularizerFactory(), alpha),
                        ),
                        graphs=(
                            # (NetworkXRegularizationGraph(graph=age_graph, name='Age'), 0.0),
                            # (NetworkXRegularizationGraph(graph=age_graph, name='Age'), beta),
                            # (NetworkXRegularizationGraph(graph=sex_graph, name='Sex_M'), 0.0),
                            (
                                NetworkXRegularizationGraph(
                                    graph=sex_graph, name="Sex_M"
                                ),
                                gamma,
                            ),
                        ),
                        regression_features=features,
                        # fitter=ADMMFitter(),
                        fitter=CVXPYFitter(),
                        warm_start=False,
                    ),
                ),
                # ('predict', LogisticRegression(fit_intercept=False))
            ]
        )
        # d[(alpha, beta, n_bins)] = evaluate_model(model)
        try:
            return -evaluate_model(model)[1]
        except:
            return 99999.9

    study: optuna.Study = optuna.create_study()
    study.enqueue_trial(
        {
            "alpha": 1e-9,
            # 'beta': 1e-9,
            "gamma": 1e-9,
            # 'nbins': 30,
        }
    )
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_value = study.best_value

    print("Best parameters:", best_params)
    print("Best value:", best_value)
    plot_slice(study)

    pass


if __name__ == "__main__":
    main()
