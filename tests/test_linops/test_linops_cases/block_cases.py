from typing import List, Tuple, Union

import numpy as np
import pytest
import pytest_cases
from scipy.linalg import block_diag

import probnum as pn
from probnum.problems.zoo.linalg import random_spd_matrix

spd_matrices = (
    pn.linops.Identity(shape=(1, 1)),
    np.array([[1.0, -2.0], [-2.0, 5.0]]),
    random_spd_matrix(np.random.default_rng(892), dim=9),
)


@pytest.mark.parametrize(
    "blocks",
    [
        ([np.array([[3, 7], [1, 5]])]),
        (
            [
                np.array([[1, 2], [3, 4]]),
                np.array(
                    [
                        [
                            5,
                            6,
                        ],
                        [0, 8],
                    ]
                ),
            ]
        ),
        ([np.array([[4, 3], [2, 1]]), 5 * np.eye(4), np.array([[2]])]),
    ],
)
@pytest_cases.case(tags=["square"])
def case_block_diagonal(
    blocks: List[np.ndarray],
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    linop = pn.linops.BlockDiagonalMatrix(*blocks)
    matrix = block_diag(*blocks)

    return linop, matrix


@pytest_cases.case(tags=["square", "symmetric", "positive-definite"])
@pytest_cases.parametrize("A", spd_matrices)
@pytest_cases.parametrize("B", spd_matrices)
@pytest_cases.parametrize("C", spd_matrices)
def case_block_diagonal_positive_definite(
    A: Union[np.ndarray, pn.linops.LinearOperator],
    B: Union[np.ndarray, pn.linops.LinearOperator],
    C: Union[np.ndarray, pn.linops.LinearOperator],
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    blocks = []
    for block in [A, B, C]:
        block = pn.linops.aslinop(block)
        block.is_symmetric = True
        block.is_positive_definite = True
        blocks.append(block)

    linop = pn.linops.BlockDiagonalMatrix(*blocks)
    matrix = block_diag(blocks[0].todense(), blocks[1].todense(), blocks[2].todense())

    return linop, matrix
