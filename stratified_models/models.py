from dataclasses import dataclass
from typing import Optional

import pandas as pd

from stratified_models.fitters.protocols import (
    NodeData,
    QuadraticStratifiedLinearRegressionProblem,
    StratifiedLinearRegressionFitter,
    Theta,
)
from stratified_models.regularization_graph.cartesian_product import (
    CartesianProductOfGraphs,
    NestedNode,
)
from stratified_models.regularization_graph.regularization_graph import (
    Node,
    RegularizationGraph,
)
from stratified_models.utils.flatten_nest import nest


@dataclass
class StratifiedLinearRegression:
    fitter: StratifiedLinearRegressionFitter[NestedNode]
    graphs: list[RegularizationGraph[Node]]
    l2_reg: float
    regression_columns: list[str]
    theta: Optional[Theta] = None

    def stratification_features(self) -> list[str]:
        return [graph.name() for graph in self.graphs]

    def get_graph(self) -> RegularizationGraph[NestedNode]:
        return CartesianProductOfGraphs.multi_product(self.graphs)

    def get_problem(
        self, x: pd.DataFrame, y: pd.Series
    ) -> QuadraticStratifiedLinearRegressionProblem:
        data = {}
        for flat_node, df_slice in x.groupby(self.stratification_features())[
            self.regression_columns
        ]:
            node = nest(flat_node) if len(self.graphs) > 1 else flat_node
            data[node] = NodeData(
                x=df_slice.values,
                y=y[df_slice.index],
            )
        graph = self.get_graph()
        return QuadraticStratifiedLinearRegressionProblem(
            nodes_data=data,
            graph=graph,
            l2_reg=self.l2_reg,
            m=len(self.regression_columns),
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        problem = self.get_problem(x=x, y=y)
        self.theta, cost = self.fitter.fit(problem=problem)
        self.theta.df.columns = self.regression_columns

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # todo: probably faster to use pd.merge (at least for large k)
        y = pd.Series(index=X.index, data=0.0)
        for z, X2 in X.groupby(self.stratification_features())[self.regression_columns]:
            y[X2.index] = X2.values @ self.theta.df.loc[z, :].values
        return y
