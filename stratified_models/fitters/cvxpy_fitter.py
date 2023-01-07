import cvxpy as cp
import networkx as nx
import pandas as pd

from stratified_models.fitters.protocols import (
    LAPLACE_REG_PARAM_KEY,
    Node,
    NodeData,
    Theta,
)
from stratified_models.utils.networkx_utils import cartesian_product


class CVXPYFitter:
    def fit(
        self,
        nodes_data: dict[Node, NodeData],
        graphs: dict[str, nx.Graph],
        l2_reg: float,
        m: int,
    ) -> Theta:
        graph = cartesian_product(graphs.values())
        problem, theta_vars = self._build_problem(
            nodes_data=nodes_data, graph=graph, l2_reg=l2_reg, m=m
        )
        problem.solve(verbose=True)  # todo: verbose flag, select solver
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise ValueError("Could not find a solution")  # todo: custom error
        theta_df = pd.DataFrame(
            {node: theta_var.value for node, theta_var in theta_vars.items()}
        ).T
        theta_df.index.names = graphs.keys()
        return theta_df

    def _build_problem(
        self,
        nodes_data: dict[Node, NodeData],
        graph: nx.Graph,
        l2_reg: float,
        m: int,
    ) -> (cp.Problem, dict[Node, cp.Variable],):
        cost = 0.0
        theta = {}

        for node in graph.nodes():
            theta[node] = cp.Variable(m)
            cost += l2_reg * cp.sum_squares(theta[node])
            if node_data := nodes_data.get(node):
                y_pred = node_data.x @ theta[node]
                cost += cp.sum_squares(y_pred - node_data.y)

        for u, v, edge_data in graph.edges(data=True):
            gamma = edge_data[LAPLACE_REG_PARAM_KEY]
            cost += gamma * cp.sum_squares(theta[u] - theta[v])
        return cp.Problem(cp.Minimize(cost)), theta
