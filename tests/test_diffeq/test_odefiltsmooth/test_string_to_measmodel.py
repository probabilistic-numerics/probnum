"""Previously, this module contained the tests for functions in the
`diffeq.odefiltsmooth.ivp2filter` module, since this module has become obsolete, we test
its replacement (`GaussianIVPFilter.string_to_measurement_model`) here.

They need different fixtures anyway.
"""

import numpy as np
import pytest

from probnum import diffeq, filtsmooth, randprocs, randvars, statespace


@pytest.fixture
def ivp():
    y0 = randvars.Constant(np.array([20.0, 15.0]))
    return diffeq.lotkavolterra([0.4124, 1.15124], y0)


@pytest.fixture
def prior(ivp):
    ode_dim = ivp.dimension
    prior = statespace.IBM(ordint=2, spatialdim=ode_dim)
    initrv = randvars.Normal(
        mean=np.zeros(prior.dimension),
        cov=np.eye(prior.dimension),
        cov_cholesky=np.eye(prior.dimension),
    )
    prior_process = randprocs.MarkovProcess(
        transition=prior, initrv=initrv, initarg=0.0
    )
    return prior_process


@pytest.mark.parametrize(
    "string, expected_type",
    [
        ("EK0", filtsmooth.DiscreteEKFComponent),
        ("EK1", filtsmooth.DiscreteEKFComponent),
    ],
)
def test_output_type(string, expected_type, ivp, prior):
    """Assert that the output type matches."""
    received = diffeq.GaussianIVPFilter.string_to_measurement_model(string, ivp, prior)
    assert isinstance(received, expected_type)


def test_string_not_supported(ivp, prior):
    with pytest.raises(ValueError):
        diffeq.GaussianIVPFilter.string_to_measurement_model("abc", ivp, prior)


@pytest.mark.parametrize(
    "string",
    ["EK0", "EK1"],
)
def test_true_mean_ek(string, ivp, prior):
    """Assert that a forwarded realization is x[1] - f(t, x[0]) with zero added covariance."""
    received = diffeq.GaussianIVPFilter.string_to_measurement_model(string, ivp, prior)
    some_real = 1.0 + 0.01 * np.random.rand(prior.transition.dimension)
    some_time = 1.0 + 0.01 * np.random.rand()
    received, _ = received.forward_realization(some_real, some_time)

    e0, e1 = prior.transition.proj2coord(0), prior.transition.proj2coord(1)
    expected = e1 @ some_real - ivp.rhs(some_time, e0 @ some_real)
    np.testing.assert_allclose(received.mean, expected)
    np.testing.assert_allclose(received.cov, 0.0, atol=1e-12)
