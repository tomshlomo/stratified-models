from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy
from numpy import typing as npt

Array = npt.NDArray[np.float64]


class LinearOperator:
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def matvec(self, x: Array) -> Array:
        # TODO: should also support matrix-matrix
        #  multiplication, or even a general tensor dot
        #  with a specified axis (which is defaulted to 0)
        raise NotImplementedError

    @abstractmethod
    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, int]:
        size = self.size()
        return size, size

    def to_scipy_linear_operator(self) -> scipy.sparse.linear.LinearOperator:
        return scipy.sparse.linalg.aslinearoperator(self)

    @property
    def dtype(self) -> type:
        return np.float64


# TODO: split to 2 classes: numpy, scipy sparse
@dataclass
class MatrixBasedLinearOperator(LinearOperator):
    a: npt.NDArray[np.float64] | scipy.sparse.spmatrix

    def size(self) -> int:
        return self.a.shape[0]

    def matvec(self, x: Array) -> Array:
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

    def matvec(self, x: Array) -> Array:
        xt = x.reshape((self.base_op.size(), -1), order="F")
        out = self.base_op.matvec(
            xt,
        )  # TODO: this assumes lin_op supports matmat which in practice is not true.
        return out.reshape(x.shape, order="F")

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

    def matvec(self, x: Array) -> Array:
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

    def matvec(self, x: Array) -> Array:
        return x

    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        return scipy.sparse.eye(self.m)


@dataclass
class SumOfLinearOperators(LinearOperator):
    components: tuple[tuple[LinearOperator, float], ...]
    m: int

    def size(self) -> int:
        return self.m

    def matvec(self, x: Array) -> Array:
        return sum(op.matvec(x) * gamma for op, gamma in self.components)  # type: ignore[return-value]

    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        return sum(op.as_sparse_matrix() * gamma for op, gamma in self.components)


@dataclass
class FlattenedTensorDot(LinearOperator):
    a: npt.NDArray[np.float64] | scipy.sparse.spmatrix  # TODO: could also be a
    # pydata.sparse array, which also supports tensordot
    axis: int
    dims: tuple[int, ...]

    def size(self) -> int:
        return int(np.prod(self.dims))

    def matvec(self, x: Array) -> Array:
        x = x.reshape(self.dims)
        # TODO: these 2 lines can be replaced with an einsum (faster)
        ax = np.tensordot(self.a, x, axes=(1, self.axis))
        ax = np.moveaxis(ax, 0, self.axis)
        return ax.ravel(order="C")

    def as_sparse_matrix(self) -> scipy.sparse.spmatrix:
        mat: scipy.sparse.spmatrix = scipy.sparse.eye(1, format="csr")
        for i, dim in enumerate(self.dims):
            if i == self.axis:
                z = (
                    self.a
                    if isinstance(self.a, scipy.sparse.spmatrix)
                    else scipy.sparse.csr_matrix(self.a)
                )
            else:
                z = scipy.sparse.eye(dim, format="csr")
            mat = scipy.sparse.kron(mat, z, format="csr")
        return mat
