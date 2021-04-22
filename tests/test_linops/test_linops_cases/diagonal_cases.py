from typing import Tuple

import numpy as np
import pytest

import probnum as pn


@pytest.mark.parametrize("n", [3, 4, 8, 12, 15])
@pytest.mark.parametrize("scalar", [0.0, 1.0, 5, -8.3])
def case_scalar_mult(n: int, scalar) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    matrix = np.diag(np.full((n,), scalar))
    return pn.linops.ScalarMult(shape=n, scalar=scalar), matrix


@pytest.mark.parametrize("n", [3, 4, 8, 12, 15])
def case_identity(n: int) -> Tuple[pn.linops.LinearOperator, np.ndarray]:
    return pn.linops.Identity(shape=n), np.eye(n)
