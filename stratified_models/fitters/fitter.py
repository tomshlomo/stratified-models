from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from stratified_models.problem import F, StratifiedLinearRegressionProblem, Theta


@dataclass
class RefitDataBase(Generic[F]):
    previous_problem: StratifiedLinearRegressionProblem[F]


# todo: fix this typing issue
RefitDataType = TypeVar("RefitDataType", bound=RefitDataBase)  # type:ignore[type-arg]


@dataclass
class ProblemUpdate:
    new_regularization_gammas: list[float]
    new_graph_gammas: list[float]

    # todo: new data? (x and y)

    def apply(
        self, problem: StratifiedLinearRegressionProblem[F]
    ) -> StratifiedLinearRegressionProblem[F]:
        return StratifiedLinearRegressionProblem(
            x=problem.x,
            y=problem.y,
            loss_factory=problem.loss_factory,
            regularizers=[
                (f, gamma)
                for (f, _), gamma in zip(
                    problem.regularizers, self.new_regularization_gammas
                )
            ],
            graphs=[
                (graph, gamma)
                for (graph, _), gamma in zip(problem.graphs, self.new_graph_gammas)
            ],
            regression_features=problem.regression_features,
        )


class Fitter(Protocol[F, RefitDataType]):
    def fit(
        self,
        problem: StratifiedLinearRegressionProblem[F],
    ) -> tuple[Theta, RefitDataType, float]:
        raise NotImplementedError

    def refit(
        self,
        problem_update: ProblemUpdate,
        refit_data: RefitDataType,
    ) -> tuple[Theta, RefitDataType, float]:
        raise NotImplementedError


# todo: wrong location


# todo: wrong location


# @dataclass
# class Costs:
#     loss: float
#     local_regularization: float
#     laplacian_regularization: float
#
#     @classmethod
#     def from_problem_and_theta(
#         cls,
#         problem: QuadraticStratifiedLinearRegressionProblem[Node],
#         theta: Theta[Node],
#     ) -> Costs:
#         loss = 0.0
#         theta_df = theta.df.set_index(pd.MultiIndex.from_tuples(theta.df.index))
#         for node, node_data in problem.nodes_data.items():
#             y_pred = node_data.x @ theta_df.loc[node, :].values
#             d = y_pred - node_data.y
#             loss += d @ d
#         local_reg = theta.df.values.ravel() @ theta.df.values.ravel() * problem.l2_reg
#         laplace_reg = problem.graph.laplacian_quad_form(theta_df.values)
#         return Costs(
#             loss=loss,
#             local_regularization=local_reg,
#             laplacian_regularization=laplace_reg,
#         )
#
#     def total(self) -> float:
#         return self.loss + self.local_regularization + self.laplacian_regularization
