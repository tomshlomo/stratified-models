from typing import Tuple

import pandas as pd
from cvxpy import quad_form
from cvxpy.atoms.sum_squares import sum_squares
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Minimize, Problem  # type: ignore

from stratified_models.fitters.protocols import Node, NodeData, Theta
from stratified_models.regularization_graph.regularization_graph import (
    Name,
    RegularizationGraph,
)


class CVXPYFitter:
    def fit(
        self,
        nodes_data: dict[Node, NodeData],
        graph: RegularizationGraph[Node, Name],
        l2_reg: float,
        m: int,
    ) -> Theta:
        problem, theta_vars = self._build_problem(
            nodes_data=nodes_data, graph=graph, l2_reg=l2_reg, m=m
        )
        # todo: verbose flag, select solver
        problem.solve(verbose=True)  # type: ignore
        theta_df = pd.DataFrame(  # todo: should be a common function
            theta_vars.value.reshape((-1, m)),
            index=graph.nodes(),
        )
        return theta_df

    def _build_problem(
        self,
        nodes_data: dict[Node, NodeData],
        graph: RegularizationGraph[Node, Name],
        l2_reg: float,
        m: int,
    ) -> Tuple[Problem, Variable]:
        k = graph.number_of_nodes()
        theta = Variable((k, m))
        cost = sum_squares(theta) * l2_reg
        for node, node_data in nodes_data.items():
            i = graph.get_node_index(node)
            y_pred = node_data.x @ theta[i]
            cost += sum_squares(y_pred - node_data.y)  # type: ignore
        laplacian = graph.laplacian_matrix()
        for i in range(m):
            cost += quad_form(theta[:, i], laplacian, assume_PSD=True) / 2
        return Problem(Minimize(cost)), theta
