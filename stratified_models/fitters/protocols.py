from dataclasses import dataclass
from typing import Protocol, TypeVar

import networkx as nx
import numpy.typing as npt
import pandas as pd

Node = TypeVar("Node")
Theta = pd.DataFrame
LAPLACE_REG_PARAM_KEY = "laplace_reg_param"


@dataclass
class NodeData:
    x: npt.NDArray
    y: npt.NDArray


class StratifiedLinearRegressionFitter(Protocol[Node]):
    def fit(
        self,
        nodes_data: dict[Node, NodeData],
        graphs: dict[str, nx.Graph],
        l2_reg: float,
        m: int,
    ) -> Theta:
        pass
