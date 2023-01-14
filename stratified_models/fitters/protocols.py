from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import TypeAlias

from stratified_models.regularization_graph.regularization_graph import (
    Name,
    Node,
    RegularizationGraph,
)

Theta: TypeAlias = pd.DataFrame


@dataclass
class NodeData:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]


class StratifiedLinearRegressionFitter(Protocol[Node, Name]):  # pragma: no cover
    def fit(
        self,
        nodes_data: dict[Node, NodeData],
        graph: RegularizationGraph[Node, Name],
        l2_reg: float,
        m: int,
    ) -> Theta:
        pass
