"""Operators with block structure."""
from __future__ import annotations

import functools

import numpy as np
from scipy.linalg import block_diag

from probnum.typing import DTypeLike, LinearOperatorLike

from . import _linear_operator, _utils


class BlockDiagonalMatrix(_linear_operator.LinearOperator):
    """Forms a block diagonal matrix from the input linear operators.

    Given square linear operators A, B, ..., Z, this represents the block
    linear operator

    .. math::
        \\begin{bmatrix}
        A & & & & \\\\
        & B & & & \\\\
        & & \\ddots & & \\\\
        & & & \\ddots & \\\\
        & & & & Z
        \\end{bmatrix}

    Parameters
    ----------
    blocks :
        The (ordered!) input linear operators. Must all be square.
    """

    def __init__(self, *blocks: LinearOperatorLike):
        blocks = tuple(_utils.aslinop(block) for block in blocks)
        if len(blocks) < 1:
            raise ValueError("At least one block must be given.")
        if not all(block.is_square for block in blocks):
            raise ValueError("All blocks must be square.")

        dtype = functools.reduce(np.promote_types, (block.dtype for block in blocks))
        shape_0 = sum(block.shape[0] for block in blocks)
        shape_1 = sum(block.shape[1] for block in blocks)
        self.split_indices = np.array(
            tuple(block.shape[1] for block in blocks)
        ).cumsum()[:-1]

        super().__init__((shape_0, shape_1), dtype)

        self.is_symmetric = all(block.is_symmetric for block in blocks)
        self.is_positive_definite = all(block.is_positive_definite for block in blocks)
        self.is_upper_triangular = all(block.is_upper_triangular for block in blocks)
        self.is_lower_triangular = all(block.is_lower_triangular for block in blocks)

        self._blocks = blocks

    @property
    def blocks(self) -> tuple[_linear_operator.LinearOperator]:
        """The linear operators that make up the diagonal (in order)."""
        return self._blocks

    def _split_input(self, x: np.ndarray) -> np.ndarray:
        return np.split(x, self.split_indices, axis=-2)

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate(
            tuple(
                cur_block @ cur_x
                for cur_block, cur_x in zip(self.blocks, self._split_input(x))
            ),
            axis=-2,
        )

    def _transpose(self) -> BlockDiagonalMatrix:
        return BlockDiagonalMatrix(*[block.T for block in self.blocks])

    def _solve(self, B: np.ndarray) -> np.ndarray:
        return np.concatenate(
            tuple(
                cur_block.solve(cur_x)
                for cur_block, cur_x in zip(self.blocks, self._split_input(B))
            ),
            axis=-2,
        )

    def _inverse(self) -> BlockDiagonalMatrix:
        return BlockDiagonalMatrix(*[block.inv() for block in self.blocks])

    def _todense(self) -> np.ndarray:
        return block_diag(*[block.todense() for block in self.blocks])

    def _det(self) -> np.inexact:
        return np.prod([block.det() for block in self.blocks])

    def _logabsdet(self) -> np.floating:
        return np.sum([block.logabsdet() for block in self.blocks])

    def _trace(self) -> np.number:
        return np.sum([block.trace() for block in self.blocks])

    def _eigvals(self) -> np.ndarray:
        return np.concatenate([block.eigvals() for block in self.blocks])

    def _rank(self) -> np.intp:
        return np.sum([block.rank() for block in self.blocks])

    def _cholesky(self, lower: bool) -> BlockDiagonalMatrix:
        return BlockDiagonalMatrix(*[block.cholesky(lower) for block in self.blocks])

    def _astype(
        self, dtype: DTypeLike, order: str, casting: str, copy: bool
    ) -> BlockDiagonalMatrix:
        return BlockDiagonalMatrix(
            *[
                block.astype(dtype, order=order, casting=casting, copy=copy)
                for block in self.blocks
            ]
        )
