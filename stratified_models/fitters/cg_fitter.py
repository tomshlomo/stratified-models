from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import scipy

from stratified_models.fitters.protocols import StratifiedLinearRegressionProblem, Theta
from stratified_models.regularization_graph.regularization_graph import Node


@dataclass
class CGFitter:
    tol: float = 1e-6
    max_iter: Optional[int] = None

    def fit(
        self,
        problem: StratifiedLinearRegressionProblem[Node],
    ) -> tuple[Theta[Node], float]:
        a, c, d = problem.build_a_c_d()
        a = a.as_scipy_linear_operator()
        c = c.ravel()
        # todo: precondition
        theta, info = scipy.sparse.linalg.cg(
            a,
            c,
            tol=self.tol,
            maxiter=self.max_iter,
        )
        cost = d - c @ theta
        theta_df = pd.DataFrame(  # todo: should be a common function
            theta.reshape((-1, problem.m)),
            index=problem.graph.nodes,
        )
        return Theta(df=theta_df), cost
