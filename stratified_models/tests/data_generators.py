from dataclasses import dataclass, field
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy

from stratified_models.fitters.admm_fitter2 import ADMMFitter2
from stratified_models.fitters.direct_fitter import DirectFitter
from stratified_models.fitters.protocols import Costs
from stratified_models.models import StratifiedLinearRegression
from stratified_models.regularization_graph.networkx_graph import (
    NetworkXRegularizationGraph,
)
from stratified_models.utils.networkx_utils import cartesian_product

RNG = np.random.default_rng(42)


@dataclass
class DataGenerator:
    n: int = 100_000
    m: int = 10
    graphs: list[Tuple[nx.Graph, float]] = field(
        default_factory=lambda: [
            # (nx.cycle_graph(200), 0.9),
            (nx.path_graph(10), 0.5),
            (nx.path_graph(12), 0.1),
        ]
    )
    smoothing_iters: int = 20
    sigma: float = 1e-1

    def regression_features(self) -> list[str]:
        return [f"x_{i}" for i in range(self.m)]

    def stratification_features(self) -> list[str]:
        return [f"z_{i}" for i in range(len(self.graphs))]

    def get_theta(self) -> pd.DataFrame:
        k_list = np.array([graph.number_of_nodes() for graph, _ in self.graphs])
        k = int(np.prod(k_list))
        regression_cols = self.regression_features()
        theta = RNG.standard_normal((k, self.m))
        for graph, weight in self.graphs:
            nx.set_edge_attributes(graph, weight, "weight")
        graph = cartesian_product(graph for graph, _ in self.graphs)
        a = nx.adjacency_matrix(graph)
        a += scipy.sparse.eye(k)
        a = scipy.sparse.diags(1 / a.sum(axis=0).A.ravel()) * a
        for _ in range(self.smoothing_iters):
            theta = a @ theta
        return pd.DataFrame(
            theta, index=pd.MultiIndex.from_tuples(graph.nodes), columns=regression_cols
        )

    def get_y(self, df: pd.DataFrame, theta: pd.DataFrame) -> pd.Series:
        regression_cols = self.regression_features()
        strat_cols = self.stratification_features()
        theta_aligned = df.loc[:, strat_cols].merge(
            theta,
            how="left",
            left_on=strat_cols,
            right_index=True,
        )
        y = (df.loc[:, regression_cols] * theta_aligned.loc[:, regression_cols]).sum(
            axis=1
        )
        y += RNG.standard_normal(self.n) * self.sigma
        return y

    def get_df(self) -> pd.DataFrame:
        regression_cols = self.regression_features()
        strat_cols = self.stratification_features()
        df = pd.DataFrame(
            RNG.standard_normal((self.n, self.m)), columns=regression_cols
        )
        for (graph, _), col in zip(self.graphs, strat_cols):
            df[col] = RNG.integers(0, graph.number_of_nodes(), self.n)
        return df

    def generate(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        theta = self.get_theta()
        df = self.get_df()
        y = self.get_y(df, theta)
        return df, y, theta


if __name__ == "__main__":
    gen = DataGenerator()
    df, y, theta_true = gen.generate()
    graphs = []
    for (graph, weight), name in zip(gen.graphs, gen.stratification_features()):
        reg = 1 / (1 - weight)
        nx.set_edge_attributes(
            graph, reg, NetworkXRegularizationGraph.LAPLACE_REG_PARAM_KEY
        )
        graphs.append(NetworkXRegularizationGraph(graph, name))
    fitter = ADMMFitter2()
    model = StratifiedLinearRegression(
        fitter=fitter,
        graphs=graphs,
        l2_reg=100,
        regression_columns=gen.regression_features(),
    )
    model.fit(df, y)
    model2 = StratifiedLinearRegression(
        fitter=DirectFitter(),
        graphs=graphs,
        l2_reg=100,
        regression_columns=gen.regression_features(),
    )
    model2.fit(df, y)
    cost = Costs.from_problem_and_theta(model.get_problem(df, y), model.theta)
    cost2 = Costs.from_problem_and_theta(model.get_problem(df, y), model2.theta)
    pass
