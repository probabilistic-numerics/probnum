"""Test fixtures for the linalg subpackage."""

import numpy as np
import pytest

from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix


@pytest.fixture(params=[pytest.param(n, id=f"dim{n}") for n in [5, 10, 50, 100]])
def n(dim) -> int:
    """Number of columns of the system matrix.

    This is mostly used for test parameterization.
    """
    return dim.param


@pytest.fixture(params=[pytest.param(seed, id=f"seed{seed}") for seed in range(5)])
def random_state(seed) -> np.random.RandomState:
    """Random states used to sample the test case input matrices.

    This is mostly used for test parameterization.
    """
    return np.random.RandomState(seed=seed.param)


@pytest.fixture()
def A_spd(n: int, random_state: np.random.RandomState):
    """Random symmetric positive definite matrix of dimension :func:`n`, sampled from
    :func:`random_state`."""
    return random_spd_matrix(dim=n, random_state=random_state)


@pytest.fixture(
    params=[
        pytest.param(density, id=f"density{density}") for density in (0.001, 0.01, 0.1)
    ]
)
def A_sparse_spd(density, n: int, random_state: np.random.RandomState):
    """Random sparse symmetric positive definite matrix of dimension :func:`n`, sampled
    from :func:`random_state`."""
    return random_sparse_spd_matrix(
        dim=n, random_state=random_state, density=density.param
    )
