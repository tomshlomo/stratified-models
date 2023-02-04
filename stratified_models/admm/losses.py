from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from stratified_models.admm.admm import Var

Matrix = npt.NDArray[np.float64]
Vector = npt.NDArray[np.float64]


@dataclass
class SumOfSquaresLoss:
    a: Matrix
    b: Vector

    def eval(self, x: Var) -> float:
        y = self.a @ x
        r = y - self.b
        return (r @ r) / 2

    def prox(self, v: Var, rho: float) -> Var:
        """
        argmin 1/2 |ax-b|^2 + rho/2 |x - v|^2
        a'(ax-b) + rho (x -v) = 0
        (a'a + rho I) x = rho v + a'b
        (a'a/rho + I) x = v + a'b/rho
        """
        m = self.a.shape[1]
        rhs = v + self.a.T @ self.b / rho
        q = self.a.T @ self.a + np.eye(m)
        return np.linalg.solve(q, rhs)
