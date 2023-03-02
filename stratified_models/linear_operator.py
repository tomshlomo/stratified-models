from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy
from numpy import typing as npt

Vector = npt.NDArray[np.float64]
NumpyArray = npt.NDArray[np.float64]


class LinearOperator:
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def matvec(self, x: Vector) -> Vector:
        # todo: should also support matrix-matrix
        #  multiplication, or even a general tensor dot
        #  with a specified axis (which is defaulted to 0)
        pass

    @abstractmethod
    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        pass

    @property
    def shape(self) -> tuple[int, int]:
        size = self.size()
        return size, size

    def to_scipy_linear_operator(self) -> scipy.sparse.linear.LinearOperator:
        return scipy.sparse.linalg.aslinearoperator(self)

    @property
    def dtype(self):
        return np.float64


# todo: split to 2 classes: numpy, scipy sparse
@dataclass
class MatrixBasedLinearOperator(LinearOperator):
    a: npt.NDArray[np.float64] | scipy.sparse.spmatrix

    def size(self) -> int:
        return self.a.shape[0]

    def matvec(self, x: Vector) -> Vector:
        return self.a @ x

    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        return (
            self.a
            if isinstance(self.a, scipy.sparse.spmatrix)
            else scipy.sparse.csr_matrix(self.a)
        )

    def to_scipy_linear_operator(self) -> scipy.sparse.linear.LinearOperator:
        return scipy.sparse.linalg.aslinearoperator(self.a)


@dataclass
class RepeatedLinearOperator(LinearOperator):
    base_op: LinearOperator
    repetitions: int

    def size(self) -> int:
        return self.base_op.size() * self.repetitions

    def matvec(self, x: Vector) -> Vector:
        xt = x.reshape((self.base_op.size(), -1), order="F")
        out = self.base_op.matvec(
            xt
        )  # todo: this assumes lin_op supports matmat which in practice is not true.
        return out.reshape(x.shape, order="F")
        # return np.repeat(out, self.repetitions, axis=0)

    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        return scipy.sparse.kron(
            scipy.sparse.eye(self.repetitions),
            self.base_op.as_sparse_matrix(),
        )


@dataclass
class BlockDiagonalLinearOperator(LinearOperator):
    blocks: dict[int, LinearOperator]
    k: int
    m: int

    def size(self) -> int:
        return self.m * self.k

    def matvec(self, x: Vector) -> Vector:
        out = np.zeros(x.shape)
        for i, block in self.blocks.items():
            slice_ = slice(i * self.m, (i + 1) * self.m)
            out[slice_] = block.matvec(x[slice_])
        return out

    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        blocks = []
        for i in range(self.k):
            if i in self.blocks:
                blocks.append(self.blocks[i].as_sparse_matrix())
            else:
                blocks.append(scipy.sparse.csr_matrix((self.m, self.m)))
        return scipy.sparse.block_diag(blocks)


@dataclass
class Identity(LinearOperator):
    m: int

    def size(self) -> int:
        return self.m

    def matvec(self, x: Vector) -> Vector:
        return x

    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        return scipy.sparse.eye(self.m)


@dataclass
class SumOfLinearOperators(LinearOperator):
    components: list[tuple[LinearOperator, float]]
    m: int

    def size(self) -> int:
        return self.m

    def matvec(self, x: Vector) -> Vector:
        return sum(op.matvec(x) * gamma for op, gamma in self.components)

    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        return sum(op.as_sparse_matrix() * gamma for op, gamma in self.components)


@dataclass
class FlattenedTensorDot(LinearOperator):
    a: npt.NDArray[np.float64] | scipy.sparse.spmatrix  # todo: could also be a
    # pydata.sparse array, which also supports tensordot
    axis: int
    dims: tuple[int, ...]

    def size(self) -> int:
        return int(np.prod(self.dims))

    def matvec(self, x: Vector) -> Vector:
        x = x.reshape(self.dims)
        ax = np.tensordot(self.a, x, axes=(1, self.axis))
        ax = ax.swapaxes(0, self.axis)
        return ax.ravel()

    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        mat = 1.0
        for i, dim in list(enumerate(self.dims)):
            z = self.a if i == self.axis else scipy.sparse.eye(dim)
            mat = scipy.sparse.kron(mat, z)
        return mat
        # size = self.size()
        # x = np.eye(size)
        # mat2 = np.zeros((size, size))
        # for i in range(size):
        #     mat2[:, i] = self.matvec(x[:, i])
        # return mat2
