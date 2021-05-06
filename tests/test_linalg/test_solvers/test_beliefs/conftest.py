"""Test fixtures for beliefs over quantities of interest of a linear system."""

import pytest

from probnum import linops, randvars
from probnum.linalg.solvers import beliefs
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix

# pylint: disable="invalid-name"


@pytest.fixture(
    params=[
        pytest.param(inv, id=inv[0])
        for inv in [
            (
                "weakmeancorr_scalar",
                beliefs.WeakMeanCorrespondenceBelief,
                lambda n: linops.Scaling(factors=1.0, shape=(n, n)),
            ),
            (
                "symmnormal_dense",
                beliefs.SymmetricNormalLinearSystemBelief,
                lambda n: randvars.Normal(
                    mean=random_spd_matrix(n, random_state=42),
                    cov=linops.SymmetricKronecker(
                        A=random_spd_matrix(n, random_state=1)
                    ),
                ),
            ),
            (
                "symmnormal_sparse",
                beliefs.SymmetricNormalLinearSystemBelief,
                lambda n: randvars.Normal(
                    mean=random_sparse_spd_matrix(n, density=0.01, random_state=42),
                    cov=linops.SymmetricKronecker(
                        A=random_sparse_spd_matrix(n, density=0.01, random_state=1)
                    ),
                ),
            ),
        ]
    ],
    name="symm_belief_multiple_rhs",
)
def fixture_symm_belief_multiple_rhs(
    request, n: int, linsys_spd_multiple_rhs: LinearSystem
) -> beliefs.SymmetricNormalLinearSystemBelief:
    """Symmetric normal linear system beliefs modelling multiple right hand sides."""
    return request.param[1].from_inverse(
        Ainv0=request.param[2](n), problem=linsys_spd_multiple_rhs
    )


@pytest.fixture(
    params=[pytest.param(alpha, id=f"alpha{alpha}") for alpha in [0.1, 1.0, 10]]
)
def scalar_weakmeancorr_prior(
    request,
    linsys_spd: LinearSystem,
) -> beliefs.WeakMeanCorrespondenceBelief:
    """Scalar weak mean correspondence belief."""
    return beliefs.WeakMeanCorrespondenceBelief.from_scalar(
        scalar=request.param, problem=linsys_spd
    )
