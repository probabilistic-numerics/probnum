from typing import Tuple

import numpy as np
import pytest

import probnum as pn


@pytest.mark.parametrize(
    "diagonal",
    [
        np.array([1.0, 2, -3]),
        np.array([1.0, 2.0, 0.0, -1.0]),
    ],
)
def case_anisotropic_scaling(
    diagonal: np.ndarray,
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Scaling(diagonal), np.diag(diagonal)


@pytest.mark.parametrize("n", [3, 4, 8, 12, 15])
@pytest.mark.parametrize(
    "scalar",
    [
        0.0,  # Singular matrix
        1.0,  # Identity
        5,  # Integer dtype
        -8.3,
    ],
)
def case_isotropic_scaling(
    n: int, scalar
) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Scaling(scalar, shape=n), np.diag(np.full((n,), scalar))
