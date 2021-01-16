"""Test fixtures for linear system beliefs."""

import pytest

import probnum.linalg.linearsolvers.beliefs as beliefs
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix


@pytest.fixture(
    params=[
        pytest.param(inv, id=inv[0])
        for inv in [
            (
                "weakmeancorr_scalar",
                beliefs.WeakMeanCorrespondenceBelief,
                lambda n: linops.ScalarMult(scalar=1.0, shape=(n, n)),
            ),
            (
                "symmnormal_dense",
                beliefs.SymmetricLinearSystemBelief,
                lambda n: rvs.Normal(
                    mean=random_spd_matrix(n, random_state=42),
                    cov=linops.SymmetricKronecker(
                        A=random_spd_matrix(n, random_state=1)
                    ),
                ),
            ),
            (
                "symmnormal_sparse",
                beliefs.SymmetricLinearSystemBelief,
                lambda n: rvs.Normal(
                    mean=random_sparse_spd_matrix(n, density=0.01, random_state=42),
                    cov=linops.SymmetricKronecker(
                        A=random_sparse_spd_matrix(n, density=0.01, random_state=1)
                    ),
                ),
            ),
        ]
    ],
    name="symm_belief",
)
def fixture_symm_belief(
    request, n: int, linsys_spd: LinearSystem
) -> beliefs.SymmetricLinearSystemBelief:
    """Symmetric normal linear system belief."""
    return request.param[1].from_inverse(Ainv0=request.param[2](n), problem=linsys_spd)
