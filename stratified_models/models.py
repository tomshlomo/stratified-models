from dataclasses import dataclass
from typing import Generic

import networkx as nx
import pandas as pd

from stratified_models.fitters.protocols import (
    Node,
    NodeData,
    StratifiedLinearRegressionFitter,
)


@dataclass
class StratifiedLinearRegression(Generic[Node]):
    fitter: StratifiedLinearRegressionFitter[Node]
    graphs: dict[str, nx.Graph]
    l2_reg: float
    regression_columns: list[str]

    def stratification_features(self) -> list[str]:
        return list(self.graphs)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        data = {
            node: NodeData(
                x=df_slice.values,
                y=y[df_slice.index],
            )
            for node, df_slice in x.groupby(self.stratification_features())[
                self.regression_columns
            ]
        }
        self.fitter.fit(
            nodes_data=data,
            graphs=self.graphs,
            l2_reg=self.l2_reg,
            m=len(self.regression_columns),
        )
