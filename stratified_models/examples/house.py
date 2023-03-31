import networkx as nx
import numpy as np
import pandas as pd
import plotly.io as pio
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from stratified_models.admm.admm import ConsensusADMMSolver
from stratified_models.estimator import StratifiedLinearEstimator
from stratified_models.fitters.admm_fitter import ADMMFitter
from stratified_models.losses import SumOfSquaresLossFactory
from stratified_models.regularization_graph.networkx_graph import (
    NetworkXRegularizationGraph,
)
from stratified_models.scalar_function import SumOfSquares

pio.renderers.default = "browser"

df = pd.read_csv(
    "stratified_models/examples/data/kc_house_data.csv",
    usecols=[
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
        "grade",
        "yr_built",
        "lat",
        "long",
    ],
)
df["log_price"] = np.log(df["price"])
df = df.query("long <= -121.6")
bins = 50
df["lat_bin"] = pd.cut(df["lat"], bins=bins)
df["long_bin"] = pd.cut(df["long"], bins=bins)
code_to_latbin = dict(enumerate(df["lat_bin"].cat.categories))
code_to_longbin = dict(enumerate(df["long_bin"].cat.categories))
df["lat_bin"] = df["lat_bin"].cat.codes
df["long_bin"] = df["long_bin"].cat.codes
df.drop(["price"], axis=1, inplace=True)
df["one"] = 1.0

df_train, df_test = model_selection.train_test_split(df, random_state=42)

regression_features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
    "grade",
    "yr_built",
    "one",
]
ss = ColumnTransformer(
    [
        (
            "scaler",
            StandardScaler().set_output(transform="pandas"),
            regression_features[:-1],
        )
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")
df_train = ss.fit_transform(df_train)
df_test = ss.transform(df_test)

common_model = LinearRegression(fit_intercept=True).fit(
    df_train[regression_features[:-1]], df_train["log_price"]
)
y_pred_test = common_model.predict(df_test[regression_features[:-1]])
y_pred_train = common_model.predict(df_train[regression_features[:-1]])
pass


def eval_pred(y, y_pred, name):
    rms = float(np.sqrt(((y - y_pred) ** 2).mean()))
    print(name, rms)


eval_pred(y_pred_test, df_test["log_price"], "common test")
eval_pred(y_pred_train, df_train["log_price"], "common train")
pass

model = StratifiedLinearEstimator(
    loss_factory=SumOfSquaresLossFactory(),
    regularizers=[(SumOfSquares(9), 1e-4)],
    graphs=[
        (NetworkXRegularizationGraph(nx.path_graph(50), "lat_bin"), 15),
        (NetworkXRegularizationGraph(nx.path_graph(50), "long_bin"), 15),
    ],
    regression_features=regression_features,
    fitter=ADMMFitter(ConsensusADMMSolver(max_iterations=1000)),
)
model.fit(
    X=df_train.drop(columns=["log_price"]),
    y=df_train["log_price"],
)
y_pred_test = model.predict(df_test)
y_pred_train = model.predict(df_train)
eval_pred(y_pred_test, df_test["log_price"], "strat test")
eval_pred(y_pred_train, df_train["log_price"], "strat train")
pass
