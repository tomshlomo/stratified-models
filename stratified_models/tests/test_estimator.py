from stratified_models.estimator import StratifiedLinearEstimator
from stratified_models.fitters.admm_fitter import ADMMFitter
from stratified_models.regularization_graph.networkx_graph import (
    NetworkXRegularizationGraph,
)
from stratified_models.tests.data_generators import DataGenerator


def test_estimator() -> None:
    gen = DataGenerator()
    df_train, y_train, df_test, y_test, theta_true = gen.generate(
        n_train=10_000,
        n_test=1000,
    )

    estimator = StratifiedLinearEstimator.make_ridge(
        gamma=10,
        graphs=tuple(
            (
                NetworkXRegularizationGraph(graph=graph, name=name),
                1 / (1 - w),
            )  # todo: 1/(1-w) is arbitrary
            for (graph, w), name in zip(gen.graphs, gen.stratification_features())
        ),
        regression_features=gen.regression_features(),
        fitter=ADMMFitter(),
        warm_start=True,
    )
    estimator.fit(X=df_train, y=y_train)
    for df, y in [(df_train, y_train), (df_test, y_test)]:
        y_pred = estimator.predict(df)
        rms = ((y_pred - y) ** 2).mean()
        assert rms <= gen.sigma * 1.1

    # refit
    reg, gamma = estimator.regularizers_factories[0]
    estimator.regularizers_factories = ((reg, gamma / 100),)
    estimator.fit(X=df_train, y=y_train)
    for df, y in [(df_train, y_train), (df_test, y_test)]:
        y_pred = estimator.predict(df)
        rms = ((y_pred - y) ** 2).mean()
        assert rms <= gen.sigma * 1.1
