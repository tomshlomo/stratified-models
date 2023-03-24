import numpy as np

from stratified_models.linear_operator import MatrixBasedLinearOperator
from stratified_models.quadratic import ExplicitQuadraticFunction


def test_explicit_quadratic() -> None:
    m = 5
    a = np.arange(m**2).reshape((m, m))
    a = a * a.T
    q = MatrixBasedLinearOperator(a)
    c = np.arange(m, dtype=np.float64)
    d = -170
    f = ExplicitQuadraticFunction(
        q=q,
        c=c,
        d=d,
    )
    assert f(np.zeros(m)) == d / 2
    assert f(np.ones(m)) == a.sum() / 2 + c.sum() + d / 2
