import cvxpy as cp
import numpy as np

from stratified_models.admm.admm import ConsensusADMMSolver, ConsensusProblem
from stratified_models.losses import SumOfSquaresLoss
from stratified_models.scalar_function import L1, SumOfSquares, Zero

# from stratified_models.admm.losses import SumOfSquaresLoss
# from stratified_models.admm.regularizers import MatrixQuadForm, QuadForm, SumOfSquares

RNG = np.random.RandomState(42)


def test_ridge_regression():
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
                f=[(SumOfSquaresLoss(a=a, b=b), 1), (SumOfSquares(m), gamma)],
                g=Zero(),
            ),
            cost_exp1,
        ),
        (
            ConsensusProblem(
                f=[(SumOfSquaresLoss(a=a, b=b), 1 / gamma)],
                g=SumOfSquares(m),
            ),
            cost_exp2,
        ),
    ]:
        x, cost, state = ConsensusADMMSolver().solve(
            problem=problem,
            x0=np.zeros(m),
            y0=np.zeros((problem.n, m)),
        )
        assert np.linalg.norm(x - x_exp) <= 1e-6
        assert abs(cost - cost_exp) <= 1e-6


def test_lasso():
    n = 100
    m = 50
    a = RNG.standard_normal((n, m))
    b = RNG.standard_normal((n,))
    gamma = 1e-1
    x_var = cp.Variable(m)
    cvxpy_problem = cp.Problem(
        cp.Minimize(cp.sum_squares(a @ x_var - b) / 2 * gamma + cp.norm1(x_var)),
    )
    cvxpy_problem.solve()
    x_exp = x_var.value
    cost_exp = cvxpy_problem.value

    problem = ConsensusProblem(
        f=[(SumOfSquaresLoss(a=a, b=b), gamma)],
        g=L1(),
    )
    x, cost, state = ConsensusADMMSolver().solve(
        problem=problem,
        x0=np.zeros(m),
        y0=np.zeros((problem.n, m)),
    )
    assert abs(cost - cost_exp) <= 1e-6
    assert np.linalg.norm(x - x_exp) <= 1e-6


#
# def test_tykhonov_regularization():
#     """
#     argmin 1/2 |ax - b|^2 + gamma/2 |x|^2 + x'qx/2
#     a'(ax-b) + gamma x + qx = 0
#     (a'a + gamma I + q) x = a'b
#     """
#     n = 10
#     m = 3
#     a = RNG.standard_normal((n, m))
#     b = RNG.standard_normal((n,))
#     q = RNG.standard_normal((m, m - 1))
#     q = q @ q.T
#     gamma = 10.0
#     x_exp = np.linalg.solve(a.T @ a + gamma * np.eye(m) + q, a.T @ b)
#     r = a @ x_exp - b
#     cost_exp = (r @ r + gamma * (x_exp @ x_exp) + x_exp.T @ q @ x_exp) / 2
#
#     problem = ConsensusProblem(
#         f=[SumOfSquaresLoss(a=a, b=b), SumOfSquares(gamma=gamma)],
#         g=QuadForm(q),
#         m=m,
#     )
#     x, cost = ConsensusADMMSolver().solve(
#         problem=problem,
#         x0=np.zeros(m),
#         y0=np.zeros((2, m)),
#     )
#     assert np.linalg.norm(x - x_exp) <= 1e-6
#     assert abs(cost - cost_exp) <= 1e-6
#
#
# def test_matrix_tykhonov_regularization():
#     """
#     argmin 1/2 |ax - b|^2 + gamma/2 |x|^2 + tr(mat(x)' q mat(x))/2
#     a'(ax-b) + gamma x + vec(q mat(x)) = 0
#     a'(ax-b) + gamma x + kron(q, I) x = 0
#     (a'a + gamma I + kron(q, I)) x = a'b
#     """
#     n = 4
#     m = 12
#     k = 6
#     a = RNG.standard_normal((n, m))
#     b = RNG.standard_normal((n,))
#     q = RNG.standard_normal((k, k - 1))
#     q = q @ q.T
#     kron_q_i = np.kron(q, np.eye(m // k))  #
#     gamma = 1.0
#     x_exp = np.linalg.solve(a.T @ a + gamma * np.eye(m) + kron_q_i, a.T @ b)
#     r = a @ x_exp - b
#     cost_exp = (r @ r + gamma * (x_exp @ x_exp) + x_exp.T @ kron_q_i @ x_exp) / 2
#
#     problem = ConsensusProblem(
#         f=[SumOfSquaresLoss(a=a, b=b), SumOfSquares(gamma=gamma)],
#         g=MatrixQuadForm(q),
#         m=m,
#     )
#     x, cost = ConsensusADMMSolver().solve(
#         problem=problem,
#         x0=np.zeros(m),
#         y0=np.zeros((2, m)),
#     )
#     assert np.linalg.norm(x - x_exp) <= 1e-6
#     assert abs(cost - cost_exp) <= 1e-6
