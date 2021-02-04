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
    return 3


@pytest.fixture
def spatialdim():
    return 1


@pytest.fixture
def spatialdim():
    return 2


@pytest.fixture
def rv1(ordint, spatialdim):
    dim = spatialdim * (ordint + 1)
    mean = np.random.rand(dim)
    cov = random_spd_matrix(dim)
    return pnrv.Normal(mean, cov)


@pytest.fixture
def rv2(ordint, spatialdim):
    dim = spatialdim * (ordint + 1)
    mean = np.random.rand(dim)
    cov = random_spd_matrix(dim)
    return pnrv.Normal(mean, cov)


@pytest.fixture
def rv3(ordint, spatialdim):
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
    return 1e-2 + 1e-2 * np.random.rand()


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


def test_rts_smooth_step_precon(dynamics_model, rv1, rv2, rv3, crosscov, dt):
    """Assert that preconditioning does not affect the outcome of the smoothing step."""

    smooth_without = pnfs.rts_smooth_step_classic
    smooth_with = pnfs.rts_add_precon(smooth_without)

    start = np.random.rand()
    stop = start + dt
    result_with, _ = smooth_with(
        rv1,
        rv3,
        rv2,
        crosscov,
        dynamics_model=dynamics_model,
        start=start,
        stop=stop,
    )
    result_without, _ = smooth_without(
        rv1,
        rv3,
        rv2,
        crosscov,
        dynamics_model=dynamics_model,
        start=start,
        stop=stop,
    )

    np.testing.assert_allclose(result_with.mean, result_without.mean)
    np.testing.assert_allclose(result_with.cov, result_without.cov)


def test_rts_smooth_step(dynamics_model, rv1, rv2, dt):

    start = np.random.rand()
    stop = start + dt
    predicted_rv, info = pnfs.predict_sqrt(dynamics_model, start, stop, rv1)

    result_with, _ = pnfs.rts_smooth_step_classic(
        rv1,
        predicted_rv,
        rv2,
        info["crosscov"],
        dynamics_model=dynamics_model,
        start=start,
        stop=stop,
    )
    result_without, _ = pnfs.rts_smooth_step_sqrt(
        rv1,
        predicted_rv,
        rv2,
        info["crosscov"],
        dynamics_model=dynamics_model,
        start=start,
        stop=stop,
    )

    np.testing.assert_allclose(result_with.mean, result_without.mean)
    np.testing.assert_allclose(result_with.cov, result_without.cov)


def test_predict(dynamics_model, rv1, dt):

    start = np.random.rand()
    stop = start + dt
    result_classic, _ = pnfs.predict_via_transition(dynamics_model, start, stop, rv1)
    result_sqrt, _ = pnfs.predict_sqrt(dynamics_model, start, stop, rv1)

    np.testing.assert_allclose(result_classic.mean, result_sqrt.mean)
    np.testing.assert_allclose(result_classic.cov, result_sqrt.cov)


@pytest.fixture
def measurement_model(ordint, spatialdim):
    H = np.random.rand(spatialdim, spatialdim * (ordint + 1))
    s = np.random.rand(spatialdim)
    Q = random_spd_matrix(spatialdim)
    return pnfss.DiscreteLTIGaussian(H, s, Q)


def test_measure(measurement_model, rv1):

    start = np.random.rand()
    result_classic, _ = pnfs.measure_via_transition(measurement_model, rv1, start)
    result_sqrt, _ = pnfs.measure_sqrt(measurement_model, rv1, start)

    np.testing.assert_allclose(result_classic.mean, result_sqrt.mean)
    np.testing.assert_allclose(result_classic.cov, result_sqrt.cov)


def test_update(measurement_model, rv1, spatialdim):
    data = np.random.rand(spatialdim)
    start = np.random.rand()
    result_classic, _, _ = pnfs.update_classic(measurement_model, rv1, start, data)
    result_sqrt, _, _ = pnfs.update_sqrt(measurement_model, rv1, start, data)

    np.testing.assert_allclose(result_classic.mean, result_sqrt.mean)
    np.testing.assert_allclose(result_classic.cov, result_sqrt.cov)
