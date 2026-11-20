from __future__ import annotations

import attrs
import structlog

from stratified_models.admm import ADMMState, ConsensusADMM, ConsensusProblem
from stratified_models.model import StartifiedModel, ThetaShape
from stratified_models.problem import (
    AbstractProblem,
    Hyperparameters,
    Solver,
)
from stratified_models.scalar_function import (
    Proxable,
    SeparableProxableScalarFunction,
    Zero,
)

logger = structlog.get_logger()


@attrs.frozen(kw_only=True)
class CompiledADMMProblem:
    loss: Proxable
    regularizers: dict[str, Proxable]
    laplacians: dict[str, Proxable]
    var_shape: tuple[int, ...]
    theta_shape: ThetaShape
    g: Proxable


@attrs.frozen(kw_only=True)
class ADMMSolver(Solver[CompiledADMMProblem, ADMMState]):
    admm: ConsensusADMM = attrs.field(factory=ConsensusADMM)

    def compile(self, abstract_problem: AbstractProblem) -> CompiledADMMProblem:
        # Build loss
        losses = []
        for node, loss in abstract_problem.group_losses():
            losses.append((abstract_problem.theta_shape.node_to_index(node), loss))

        separable_loss = SeparableProxableScalarFunction(items=tuple(losses))

        # Build regularizers
        regularizers = {}
        for name, reg in abstract_problem.regularizers.items():
            if isinstance(reg, Proxable):
                regularizers[name] = reg
            else:
                # If not proxable, we can't use ADMM easily unless we wrap it or error
                # For now assume all are proxable
                pass

        # Build laplacians
        laplacians = {
            name: lap
            for name, lap in abstract_problem.laplacians()
            if isinstance(lap, Proxable)
        }

        return CompiledADMMProblem(
            loss=separable_loss,
            regularizers=regularizers,
            laplacians=laplacians,
            var_shape=abstract_problem.theta_shape.array_shape,
            theta_shape=abstract_problem.theta_shape,
            g=Zero(),  # Domain constraints if any
        )

    def solve(
        self,
        compiled_problem: CompiledADMMProblem,
        hyperparameters: Hyperparameters,
    ) -> tuple[StartifiedModel, ADMMState]:
        f: list[tuple[Proxable, float]] = [(compiled_problem.loss, 1.0)]

        for name, reg in compiled_problem.regularizers.items():
            gamma = hyperparameters.regularizers.get(name, 0.0)
            if gamma > 0:
                f.append((reg, gamma))

        for name, lap in compiled_problem.laplacians.items():
            gamma = hyperparameters.graphs.get(name, 0.0)
            if gamma > 0:
                f.append((lap, gamma))

        problem = ConsensusProblem(
            f=tuple(f),
            g=compiled_problem.g,
            var_shape=compiled_problem.var_shape,
        )

        best_z, _, state = self.admm.solve(problem)

        model = StartifiedModel(
            theta=best_z,
            shape=compiled_problem.theta_shape,
        )
        return model, state
