from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from stratified_models.admm.admm import Var

Matrix = npt.NDArray[np.float64]
Vector = npt.NDArray[np.float64]


@dataclass
class SumOfSquares:
    gamma: float

    def eval(self, x: Var) -> float:
        return (x @ x) * self.gamma / 2

    def prox(self, v: Var, rho: float) -> Var:
        """
        argmin gamma/2 |x|^2 + rho/2 |x - v|^2
        x gamma + (x - v) rho = 0
        x (rho + gamma) = v rho
        x = v rho / (rho + gamma)
        """
        return v * (rho / (rho + self.gamma))


@dataclass
class QuadForm:
    """
    x' q x / 2
    """

    q: Matrix

    def eval(self, x: Var) -> float:
        return (x @ (self.q @ x)) / 2

    def prox(self, v: Var, rho: float) -> Var:
        """
        argmin 1/2 x' q x + rho/2 |x - v|^2
        q x + (x - v) rho = 0
        (q + rho I) x = v rho
        (q/rho + I) x = v
        """
        return np.linalg.solve(self.q / rho + np.eye(self.q.shape[0]), v)


@dataclass
class MatrixQuadForm:
    """
    tr(mat(x)' q max(x)) / 2
    """

    q: Matrix

    def eval(self, x: Var) -> float:
        # todo: einsum
        x = x.reshape((self.q.shape[0], -1))
        qx = self.q @ x
        return np.sum(x * qx) / 2

    def prox(self, v: Var, rho: float) -> Var:
        v = v.reshape((self.q.shape[0], -1))
        return np.linalg.solve(self.q / rho + np.eye(self.q.shape[0]), v).ravel()


def soft_threshold(
    x: npt.NDArray[np.float64], thresh: float
) -> npt.NDArray[np.float64]:
    return np.clip(x - thresh, 0.0, None) - np.clip(-x - thresh, 0.0, None)


@dataclass
class L1:
    gamma: float

    def eval(self, x: Var) -> float:
        return self.gamma * np.sum(np.abs(x))

    def prox(self, v: Var, rho: float) -> Var:
        """
        argmin gamma |x| + rho/2 |x - v|^2
        """
        return soft_threshold(v, self.gamma / rho)


@dataclass
class NonNegativeIndicator:
    def eval(self, x: Var) -> float:
        return float("inf") if (x < 0).any() else 0.0

    def prox(self, v: Var, rho: float) -> Var:
        """
        argmin rho/2 |x - v|^2 s.t x>=0
        """
        return np.clip(v, 0.0, None)
