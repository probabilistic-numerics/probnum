"""Operators with block structure."""
from __future__ import annotations

import functools
from typing import Callable, Optional

import numpy as np
from scipy.linalg import block_diag

from probnum.typing import DTypeLike, LinearOperatorLike

from . import _linear_operator, _utils


class BlockDiagonalMatrix(_linear_operator.LinearOperator):
    """Forms a block diagonal matrix from the input linear operators.

    Given linear operators A, B, ..., Z, this represents the block linear
    operator

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
        The (ordered!) input linear operators.
    """

    def __init__(self, *blocks: LinearOperatorLike):
        blocks = tuple(_utils.aslinop(block) for block in blocks)
        if len(blocks) < 1:
            raise ValueError("At least one block must be given.")
        self._all_blocks_square = all(block.is_square for block in blocks)

        dtype = functools.reduce(np.promote_types, (block.dtype for block in blocks))
        shape_0 = sum(block.shape[0] for block in blocks)
        shape_1 = sum(block.shape[1] for block in blocks)
        self.split_indices = np.array(
            tuple(block.shape[1] for block in blocks)
        ).cumsum()[:-1]

        super().__init__((shape_0, shape_1), dtype)

        self._blocks = blocks

        self.is_symmetric = self._infer_property(lambda block: block.is_symmetric)
        self.is_positive_definite = self._infer_property(
            lambda block: block.is_positive_definite
        )
        self.is_upper_triangular = self._infer_property(
            lambda block: block.is_upper_triangular
        )
        self.is_lower_triangular = self._infer_property(
            lambda block: block.is_lower_triangular
        )

    @property
    def blocks(self) -> tuple[_linear_operator.LinearOperator]:
        """The linear operators that make up the diagonal (in order)."""
        return self._blocks

    def _infer_property(
        self, property_fn: Callable[[_linear_operator.LinearOperator], Optional[bool]]
    ) -> Optional[bool]:
        if all(property_fn(block) for block in self.blocks):
            return True
        if any(property_fn(block) is False for block in self.blocks):
            return False
        return None

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
        if self._all_blocks_square:
            return BlockDiagonalMatrix(*[block.inv() for block in self.blocks])
        return super()._inverse()

    def _todense(self) -> np.ndarray:
        return block_diag(*[block.todense() for block in self.blocks])

    def _det(self) -> np.inexact:
        if self._all_blocks_square:
            return np.prod([block.det() for block in self.blocks])
        return super()._det()

    def _logabsdet(self) -> np.floating:
        if self._all_blocks_square:
            return np.sum([block.logabsdet() for block in self.blocks])
        return super()._logabsdet()

    def _trace(self) -> np.number:
        if self._all_blocks_square:
            return np.sum([block.trace() for block in self.blocks])
        return super()._trace()

    def _eigvals(self) -> np.ndarray:
        if self._all_blocks_square:
            return np.concatenate([block.eigvals() for block in self.blocks])
        return super()._eigvals()

    def _rank(self) -> np.intp:
        if self._all_blocks_square:
            return np.sum([block.rank() for block in self.blocks])
        return super()._rank()

    def _cholesky(self, lower: bool) -> BlockDiagonalMatrix:
        if self._all_blocks_square:
            return BlockDiagonalMatrix(
                *[block.cholesky(lower) for block in self.blocks]
            )
        return super()._cholesky(lower)

    def _astype(
        self, dtype: DTypeLike, order: str, casting: str, copy: bool
    ) -> BlockDiagonalMatrix:
        return BlockDiagonalMatrix(
            *[
                block.astype(dtype, order=order, casting=casting, copy=copy)
                for block in self.blocks
            ]
        )
