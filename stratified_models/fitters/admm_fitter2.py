from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from stratified_models.admm.admm import (
    ConsensusADMMSolver,
    ConsensusProblem,
    SeparableSumOfProxables,
)
from stratified_models.admm.losses import SumOfSquaresLoss
from stratified_models.admm.regularizers import MatrixQuadForm, SumOfSquares
from stratified_models.fitters.protocols import (
    Costs,
    QuadraticStratifiedLinearRegressionProblem,
    Theta,
)
from stratified_models.regularization_graph.regularization_graph import (
    Node,
    RegularizationGraph,
)


@dataclass
class LaplacianWrapper:
    graph: RegularizationGraph[Node]

    def eval(self, x: npt.NDArray[np.float64]) -> float:
        x = x.reshape((self.graph.number_of_nodes(), -1))
        # out = self.graph.laplacian_quad_form(x)
        f = MatrixQuadForm(self.graph.laplacian_matrix() * 2)
        out2 = f.eval(x.ravel())
        return out2

    def prox(self, v: npt.NDArray[np.float64], rho: float) -> npt.NDArray[np.float64]:
        v = v.reshape((self.graph.number_of_nodes(), -1))
        # out = self.graph.laplacian_prox(v, rho).ravel()
        f = MatrixQuadForm(self.graph.laplacian_matrix() * 2)
        out2 = f.prox(v.ravel(), rho)
        return out2


@dataclass
class ADMMFitter2:
    tol: float = 1e-6
    max_iter: int = 1000

    def fit(
        self,
        problem: QuadraticStratifiedLinearRegressionProblem[Node],
    ) -> tuple[Theta[Node], Costs]:
        k = problem.graph.number_of_nodes()
        mk = problem.m * k

        loss = SeparableSumOfProxables(
            f={
                problem.graph.get_node_index(node): SumOfSquaresLoss(
                    a=node_data.x,
                    b=node_data.y,
                )
                for node, node_data in problem.nodes_data.items()
            },
            n=k,
            m=problem.m,
        )

        local_reg = SumOfSquares(gamma=problem.l2_reg)
        consensus_problem = ConsensusProblem(
            f=[loss, local_reg],
            g=LaplacianWrapper(problem.graph),
            # g=MatrixQuadForm(problem.graph.laplacian_matrix() * 2),
            # g=QuadForm(np.kron(
            #     problem.graph.laplacian_matrix().toarray(),
            #     np.eye(problem.m),
            # )),
            m=mk,
        )
        theta, cost = ConsensusADMMSolver().solve(
            problem=consensus_problem,
            x0=np.zeros(mk),
            y0=np.zeros((2, mk)),
        )
        theta_df = pd.DataFrame(  # todo: should be a common function
            theta.reshape((k, problem.m)),
            index=problem.graph.nodes,
        )
        return Theta(df=theta_df), cost
