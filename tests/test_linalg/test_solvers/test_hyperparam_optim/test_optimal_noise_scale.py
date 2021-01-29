"""Tests for the optimal noise scale of noisy linear systems."""
import pytest

from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.linalg.solvers.data import LinearSolverData
from probnum.linalg.solvers.hyperparam_optim import OptimalNoiseScale
from probnum.problems import NoisyLinearSystem


def test_noise_scale_nonnegative(optimal_noise_scale: float):
    """Test whether the estimated noise scale is non-negative."""
    assert optimal_noise_scale >= 0.0


@pytest.mark.parametrize("eps", [0.0], indirect=True)
def test_noise_free_matrix(eps: float, optimal_noise_scale: float):
    """Test whether in the exact case zero noise is estimated."""
    assert optimal_noise_scale == pytest.approx(0.0)


def test_recovers_true_noise_scale(eps: float, optimal_noise_scale: float):
    """Test whether given enough observations the true noise scale of the system is
    found."""
    assert optimal_noise_scale == pytest.approx(eps)


def test_iterative_and_batch_identical(
    eps: float,
    linsys_matnoise: NoisyLinearSystem,
    prior: LinearSystemBelief,
    noisy_solver_data: LinearSolverData,
):
    """Test whether computing the optimal noise scale iteratively matches its
    computation in batched form."""

    # Batch computed optimal noise scale
    noiseA_iter = OptimalNoiseScale(iterative=True)(
        problem=linsys_matnoise, belief=prior, data=noisy_solver_data
    )
    eps_iter = noiseA_iter.epsA_cov.A.args[1]

    # Iteratively computed optimal noise scale
    noiseA_batch = OptimalNoiseScale(iterative=False)(
        problem=linsys_matnoise, belief=prior, data=noisy_solver_data
    )
    eps_batch = noiseA_batch.epsA_cov.A.args[1]

    assert pytest.approx(eps_iter) == eps_batch
