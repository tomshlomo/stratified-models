import cvxpy as cp
import numpy as np

from stratified_models.admm.admm import ConsensusADMMSolver, ConsensusProblem
from stratified_models.losses import SumOfSquaresLoss
from stratified_models.scalar_function import (
    L1,
    NonNegativeIndicator,
    SumOfSquares,
    Zero,
)

RNG = np.random.RandomState(42)


def test_ridge_regression() -> None:
    """
    argmin 1/2 |ax - b|^2 + gamma/2 |x|^2
    a'(ax-b) + gamma x = 0
    (a'a + gamma I) x = a'b
    """
    n = 10
    m = 3
    a = RNG.standard_normal((n, m))
    b = RNG.standard_normal((n,))
    gamma = 10.0
    x_exp = np.linalg.solve(a.T @ a + gamma * np.eye(m), a.T @ b)
    r = a @ x_exp - b
    cost_exp1 = (r @ r + gamma * (x_exp @ x_exp)) / 2
    cost_exp2 = cost_exp1 / gamma

    for problem, cost_exp in [
        (
            ConsensusProblem(
                f=((SumOfSquaresLoss(a=a, b=b), 1), (SumOfSquares(m), gamma)),
                g=Zero(),
                var_shape=(m,),
            ),
            cost_exp1,
        ),
        (
            ConsensusProblem(
                f=((SumOfSquaresLoss(a=a, b=b), 1 / gamma),),
                g=SumOfSquares(m),
                var_shape=(m,),
            ),
            cost_exp2,
        ),
    ]:
        x, cost, state, converged = ConsensusADMMSolver().solve(problem=problem)
        assert np.linalg.norm(x - x_exp) <= 1e-4
        assert abs(cost - cost_exp) <= 1e-6
        assert converged

        _, _, _, converged = ConsensusADMMSolver(max_iterations=5).solve(
            problem=problem
        )
        assert not converged


def test_lasso() -> None:
    n = 100
    m = 50
    a = RNG.standard_normal((n, m))
    b = RNG.standard_normal((n,))
    gamma = 1e-1
    x_var = cp.Variable(m)  # type: ignore[attr-defined]
    cost = cp.sum_squares(  # type: ignore[attr-defined]
        a @ x_var - b
    ) / 2 * gamma + cp.norm1(  # type: ignore[attr-defined]
        x_var
    )
    cvxpy_problem = cp.Problem(  # type: ignore[attr-defined]
        cp.Minimize(cost),  # type: ignore[attr-defined]
    )
    cvxpy_problem.solve()
    x_exp = x_var.value
    cost_exp = cvxpy_problem.value

    problem = ConsensusProblem(
        f=((SumOfSquaresLoss(a=a, b=b), gamma),),
        g=L1(),
        var_shape=(m,),
    )
    x, cost, state, converged = ConsensusADMMSolver().solve(problem=problem)
    assert abs(cost - cost_exp) <= 1e-6
    assert np.linalg.norm(x - x_exp) <= 1e-4
    assert converged


def test_non_negative_least_squares() -> None:
    m = 10
    b = RNG.standard_normal((m,))
    problem = ConsensusProblem(
        f=((SumOfSquaresLoss(a=np.eye(m), b=b), 1.0),),
        g=NonNegativeIndicator(),
        var_shape=(m,),
    )
    x, cost, state, converged = ConsensusADMMSolver().solve(problem=problem)
    x_exp = b.clip(min=0)
    assert np.linalg.norm(x - x_exp) <= 1e-6
    assert converged
