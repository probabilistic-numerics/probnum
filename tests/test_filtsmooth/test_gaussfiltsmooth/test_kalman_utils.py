"""Test Kalman utility functions."""


import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv
from probnum.problems.zoo.linalg import random_spd_matrix


@pytest.fixture
def ordint():
    return 0


@pytest.fixture
def ordint():
    return 1


@pytest.fixture
def ordint():
    return 6


@pytest.fixture
def spatialdim():
    return 1


@pytest.fixture
def spatialdim():
    return 3


@pytest.fixture
def unsmoothed_rv(ordint, spatialdim):
    dim = spatialdim * (ordint + 1)
    mean = np.random.rand(dim)
    cov = random_spd_matrix(dim)
    return pnrv.Normal(mean, cov)


@pytest.fixture
def smoothed_rv(ordint, spatialdim):
    dim = spatialdim * (ordint + 1)
    mean = np.random.rand(dim)
    cov = random_spd_matrix(dim)
    return pnrv.Normal(mean, cov)


@pytest.fixture
def predicted_rv(ordint, spatialdim):
    dim = spatialdim * (ordint + 1)
    mean = np.random.rand(dim)
    cov = random_spd_matrix(dim)
    return pnrv.Normal(mean, cov)


@pytest.fixture
def crosscov(ordint, spatialdim):
    dim = spatialdim * (ordint + 1)
    crosscov = np.random.rand(dim, dim)
    return crosscov


@pytest.fixture
def dt():
    return 0.1 + 0.1 * np.random.rand()


@pytest.fixture
def dt():
    return 1e-3 + 1e-3 * np.random.rand()


@pytest.fixture
def dynamics_model(ordint, spatialdim):
    dynamics_model = pnfss.IBM(ordint, spatialdim)
    return dynamics_model


@pytest.fixture
def dynamics_model(ordint, spatialdim):
    dynamics_model = pnfss.IOUP(ordint, spatialdim, driftspeed=1.0)
    return dynamics_model


@pytest.fixture
def dynamics_model(ordint, spatialdim):
    dynamics_model = pnfss.Matern(ordint, spatialdim, lengthscale=1.0)
    return dynamics_model


def test_rts_smooth_step(
    dynamics_model, unsmoothed_rv, smoothed_rv, predicted_rv, crosscov, dt
):
    """Assert that preconditioning does not affect the outcome of the smoothing step."""
    smooth_without = pnfs.rts_smooth_step_classic
    smooth_with = pnfs.rts_add_precon(smooth_without)

    start = np.random.rand()
    stop = start + dt
    result_with, _ = smooth_with(
        unsmoothed_rv,
        predicted_rv,
        smoothed_rv,
        crosscov,
        dynamics_model=dynamics_model,
        start=start,
        stop=stop,
    )
    result_without, _ = smooth_without(
        unsmoothed_rv,
        predicted_rv,
        smoothed_rv,
        crosscov,
        dynamics_model=dynamics_model,
        start=start,
        stop=stop,
    )

    np.testing.assert_allclose(result_with.mean, result_without.mean)
