"""Previously, this module contained the tests for functions in the
`diffeq.odefiltsmooth.gaussian.ivp2filter` module, since this module has become
obsolete, we test its replacement (`GaussianIVPFilter.string_to_measurement_model`)
here.

They need different fixtures anyway.
"""

import numpy as np
import pytest

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import diffeq, filtsmooth, randprocs, randvars


@pytest.fixture
def ivp():
    y0 = np.array([20.0, 15.0])
    return diffeq_zoo.lotkavolterra(t0=0.4124, tmax=1.15124, y0=y0)


@pytest.fixture
def prior(ivp):
    ode_dim = ivp.dimension
    prior = randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=2, wiener_process_dimension=ode_dim
    )
    initrv = randvars.Normal(
        mean=np.zeros(prior.state_dimension),
        cov=np.eye(prior.state_dimension),
        cov_cholesky=np.eye(prior.state_dimension),
    )
    prior_process = randprocs.markov.MarkovProcess(
        transition=prior, initrv=initrv, initarg=0.0
    )
    return prior_process


@pytest.mark.parametrize(
    "string, expected_type",
    [
        ("EK0", filtsmooth.gaussian.approx.DiscreteEKFComponent),
        ("EK1", filtsmooth.gaussian.approx.DiscreteEKFComponent),
    ],
)
def test_output_type(string, expected_type, ivp, prior):
    """Assert that the output type matches."""
    received = diffeq.odefiltsmooth.GaussianIVPFilter.string_to_measurement_model(
        string, ivp, prior
    )
    assert isinstance(received, expected_type)


def test_string_not_supported(ivp, prior):
    with pytest.raises(ValueError):
        diffeq.odefiltsmooth.GaussianIVPFilter.string_to_measurement_model(
            "abc", ivp, prior
        )


@pytest.mark.parametrize(
    "string",
    ["EK0", "EK1"],
)
def test_true_mean_ek(string, ivp, prior):
    """Assert that a forwarded realization is x[1] - f(t, x[0]) with zero added covariance."""
    received = diffeq.odefiltsmooth.GaussianIVPFilter.string_to_measurement_model(
        string, ivp, prior
    )
    some_real = 1.0 + 0.01 * np.random.rand(prior.transition.state_dimension)
    some_time = 1.0 + 0.01 * np.random.rand()
    received, _ = received.forward_realization(some_real, some_time)

    e0, e1 = prior.transition.proj2coord(0), prior.transition.proj2coord(1)
    expected = e1 @ some_real - ivp.f(some_time, e0 @ some_real)
    np.testing.assert_allclose(received.mean, expected)
    np.testing.assert_allclose(received.cov, 0.0, atol=1e-12)
