# from dataclasses import dataclass
# from typing import Sequence
#
# import networkx as nx
# import pandas as pd
#
# from stratified_models.fitters.protocols import (
#     NodeData,
#     StratifiedLinearRegressionFitter,
# )
#
#
# @dataclass
# class StratifiedLinearRegression:
#     fitter: StratifiedLinearRegressionFitter
#     graphs: dict[str, nx.Graph]
#     l2_reg: float
#     regression_columns: list[str]
#
#     def stratification_features(self) -> Sequence[str]:
#         return self.graphs.keys()
#
#     def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
#         data = {
#             node: NodeData(
#                 x=df_slice.values,
#                 y=y[df_slice.index],
#             )
#             for node, df_slice in x.groupby(self.stratification_features()).loc[
#                 :, self.regression_columns
#             ]
#         }
#         self.fitter.fit(
#             nodes_data=data,
#             graphs=self.graphs,
#             l2_reg=self.l2_reg,
#             m=len(self.regression_columns),
#         )
