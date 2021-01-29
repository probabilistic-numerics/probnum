"""Tests for the weak mean correspondence belief update under linear observations."""
import numpy as np
import pytest

from probnum.linalg.solvers.beliefs import WeakMeanCorrespondenceBelief


def test_means_correspond_weakly(
    weakmeancorr_updated_belief: WeakMeanCorrespondenceBelief,
    matvec_observation: np.ndarray,
):
    r"""Test whether :math:`\mathbb{E}[A]^{-1}y = \mathbb{E}[H]y` for all actions
    :math:`y`."""
    np.testing.assert_allclose(
        np.linalg.solve(
            weakmeancorr_updated_belief.A.mean.todense(), matvec_observation.obsA
        ),
        weakmeancorr_updated_belief.Ainv.mean @ matvec_observation.obsA,
    )


@pytest.mark.parametrize("n", [3, 5, 10], indirect=True)
def test_iterative_covariance_trace_update(
    n: int,
    num_iters: int,
    weakmeancorr_updated_belief: WeakMeanCorrespondenceBelief,
):
    """The solver's returned value for the trace must match the actual trace of the
    solution covariance."""

    assert weakmeancorr_updated_belief.A.cov.trace() == pytest.approx(
        np.trace(weakmeancorr_updated_belief.A.cov.todense()),
    )
    assert weakmeancorr_updated_belief.Ainv.cov.trace() == pytest.approx(
        np.trace(weakmeancorr_updated_belief.Ainv.cov.todense()),
    )
