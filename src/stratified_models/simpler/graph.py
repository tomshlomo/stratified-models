from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from functools import cached_property

import attrs
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from numpy import typing as npt
from scipy.sparse import sparray

from stratified_models.simpler.model import Stratification
from stratified_models.simpler.scalar_function import (
    ScalarFunction,
    SparseQuadraticForm,
)


class RegularizationGraph(ABC):
    stratification: Stratification

    @abstractmethod
    def laplacian(self, axis: int, dims: tuple[int, ...]) -> ScalarFunction:
        pass

    def get_subnode_index(self, sub_node: Hashable) -> int:
        return self.stratification.get_subnode_index(sub_node)

    @property
    def size(self) -> int:
        return self.stratification.size

    @property
    def name(self) -> Hashable:
        return self.stratification.name


@attrs.frozen(kw_only=True)
class NetworkXRegularizationGraph(RegularizationGraph):
    stratification: Stratification
    graph: nx.Graph
    weight_key: str = "weight"

    @cached_property
    def laplacian_matrix(self) -> sparray:
        return nx.laplacian_matrix(self.graph, weight=self.weight_key)

    def laplacian(self, axis: int, dims: tuple[int, ...]) -> SparseQuadraticForm:
        return SparseQuadraticForm(
            a=self.laplacian_matrix,
            axis=axis,
            dims=dims,
        )

    @staticmethod
    def path(n: int, name: Hashable) -> NetworkXRegularizationGraph:
        graph = nx.path_graph(n)
        nx.set_edge_attributes(graph, 1.0, "weight")
        return NetworkXRegularizationGraph(
            stratification=Stratification(index=pd.Index(range(n), name=name)),
            graph=graph,
            weight_key="weight",
        )

    @staticmethod
    def voronoi(
        points: npt.NDArray[np.float64],
        name: Hashable,
    ) -> NetworkXRegularizationGraph:
        voronoi = scipy.spatial.Voronoi(points)
        graph = nx.Graph()
        for i, point in enumerate(voronoi.points):
            graph.add_node(i, point=point)
        # edges are voronoi ridges
        for edge in voronoi.ridge_points:
            graph.add_edge(edge[0], edge[1], weight=1.0)
        nx.set_edge_attributes(graph, 1.0, "weight")
        return NetworkXRegularizationGraph(
            stratification=Stratification(
                index=pd.Index(range(len(voronoi.points)), name=name)
            ),
            graph=graph,
            weight_key="weight",
        )
