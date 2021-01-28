"""Tests for the optimal noise scale of noisy linear systems."""
import pytest

from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.linalg.solvers.data import LinearSolverData
from probnum.linalg.solvers.hyperparam_optim import OptimalNoiseScale
from probnum.problems import NoisyLinearSystem


def test_learns_true_noise_scale(
    eps: float,
    linsys_matnoise: NoisyLinearSystem,
    optimal_noise_scale: OptimalNoiseScale,
    prior: LinearSystemBelief,
    noisy_solver_data: LinearSolverData,
):
    """Test whether given enough observations the true noise scale of the system is
    found."""
    noiseA = optimal_noise_scale(
        problem=linsys_matnoise, belief=prior, data=noisy_solver_data
    )
    eps_est = noiseA.epsA_cov.A.args[1]

    assert eps_est == pytest.approx(eps)


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
