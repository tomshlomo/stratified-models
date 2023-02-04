import pandas as pd
import scipy

from stratified_models.fitters.protocols import (
    QuadraticStratifiedLinearRegressionProblem,
    Theta,
)
from stratified_models.regularization_graph.regularization_graph import Node


class DirectFitter:
    def fit(
        self,
        problem: QuadraticStratifiedLinearRegressionProblem[Node],
    ) -> tuple[Theta[Node], float]:
        a, c, d = problem.build_a_c_d()
        a = a.as_matrix()
        c = c.ravel()
        theta = scipy.sparse.linalg.spsolve(a, c)
        cost = d - c @ theta
        theta = theta.reshape((-1, problem.m))
        theta_df = pd.DataFrame(
            theta,
            index=problem.graph.nodes,
        )
        return Theta(df=theta_df), cost
