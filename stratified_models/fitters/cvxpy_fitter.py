import pandas as pd
from cvxpy import quad_form
from cvxpy.atoms.sum_squares import sum_squares
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Minimize, Problem  # type: ignore

from stratified_models.fitters.protocols import (
    QuadraticStratifiedLinearRegressionProblem,
    Theta,
)
from stratified_models.regularization_graph.regularization_graph import Node


class CVXPYFitter:
    def fit(
        self,
        problem: QuadraticStratifiedLinearRegressionProblem,
    ) -> tuple[Theta[Node], float]:
        cvxpy_problem, theta_vars = self._build_cvxpy_problem(problem)
        # todo: verbose flag, select solver
        cvxpy_problem.solve(verbose=True)  # type: ignore
        theta_df = pd.DataFrame(  # todo: should be a common function
            theta_vars.value.reshape((-1, problem.m)),
            index=problem.graph.nodes,
        )
        return Theta(df=theta_df), cvxpy_problem.objective.value

    def _build_cvxpy_problem(
        self,
        problem: QuadraticStratifiedLinearRegressionProblem,
    ) -> tuple[Problem, Variable]:
        k = problem.graph.number_of_nodes()
        theta = Variable((k, problem.m))
        local_regularization = sum_squares(theta) * problem.l2_reg
        loss = 0.0
        for node, node_data in problem.nodes_data.items():
            i = problem.graph.get_node_index(node)
            y_pred = node_data.x @ theta[i]
            loss += sum_squares(y_pred - node_data.y)  # type: ignore[no-untyped-call]
        laplacian = problem.graph.laplacian_matrix()
        laplacian_regularization = 0.0
        for i in range(problem.m):
            laplacian_regularization += quad_form(
                theta[:, i], laplacian, assume_PSD=True
            )
        cost = loss + local_regularization + laplacian_regularization
        return Problem(Minimize(cost)), theta
