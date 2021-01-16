"""Tests for the weak mean correspondence belief update under linear observations."""
import numpy as np
import pytest

from probnum.linalg.linearsolvers.belief_updates import (
    WeakMeanCorrLinearObsBeliefUpdate,
)

pytestmark = pytest.mark.usefixtures("weakmeancorrlin_belief_update")


def test_means_correspond_weakly(
    weakmeancorrlin_belief_update: WeakMeanCorrLinearObsBeliefUpdate,
    matvec_observation: np.ndarray,
):
    r"""Test whether :math:`\mathbb{E}[A]^{-1}y = \mathbb{E}[H]y` for all actions
    :math:`y`."""
    np.testing.assert_allclose(
        np.linalg.solve(
            weakmeancorrlin_belief_update.A.mean.todense(), matvec_observation
        ),
        weakmeancorrlin_belief_update.Ainv.mean @ matvec_observation,
    )


@pytest.mark.parametrize("n", [3, 5, 10], indirect=True)
def test_iterative_covariance_trace_update(
    n: int,
    weakmeancorrlin_belief_update: WeakMeanCorrLinearObsBeliefUpdate,
):
    """The solver's returned value for the trace must match the actual trace of the
    solution covariance."""
    pytest.approx(
        weakmeancorrlin_belief_update.A.cov.trace(),
        np.trace(weakmeancorrlin_belief_update.A.cov.todense()),
    )
    pytest.approx(
        weakmeancorrlin_belief_update.Ainv.cov.trace(),
        np.trace(weakmeancorrlin_belief_update.Ainv.cov.todense()),
    )
