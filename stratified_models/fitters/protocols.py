from dataclasses import dataclass
from typing import Protocol, TypeVar

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import TypeAlias

Node = TypeVar("Node")
Theta: TypeAlias = pd.DataFrame
LAPLACE_REG_PARAM_KEY = "laplace_reg_param"


@dataclass
class NodeData:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]


class StratifiedLinearRegressionFitter(Protocol[Node]):  # pragma: no cover
    def fit(
        self,
        nodes_data: dict[Node, NodeData],
        graphs: dict[str, nx.Graph],
        l2_reg: float,
        m: int,
    ) -> Theta:
        pass
