"""Test cases for Bayesian quadrature."""

import numpy as np
import pytest
from scipy.integrate import quad

from probnum.quad import BayesianQuadrature, sample_from_measure
from probnum.randvars import Normal


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_from_measure_policy(rng):
    """Fix the random number generator in the "sample_from_measure" policy, thus make
    this policy fit the common interface."""

    def policy(nevals, measure):
        return sample_from_measure(rng=rng, nevals=nevals, measure=measure)

    return policy


@pytest.mark.parametrize("input_dim", [1], ids=["dim1"])
def test_type_1d(f1d, kernel, measure, sample_from_measure_policy):
    """Test that BQ outputs normal random variables for 1D integrands."""
    # pylint: disable=invalid-name
    bq = BayesianQuadrature(kernel=kernel, policy=sample_from_measure_policy)
    integral, _ = bq.integrate(fun=f1d, measure=measure, nevals=10)
    assert isinstance(integral, Normal)


@pytest.mark.parametrize("input_dim", [1])
def test_integral_values_1d(f1d, kernel, measure, sample_from_measure_policy):
    """Test numerically that BQ computes 1D integrals correctly."""

    # numerical integral
    # pylint: disable=invalid-name
    def integrand(x):
        return f1d(x) * measure(x)

    # pylint: disable=invalid-name
    bq = BayesianQuadrature(kernel=kernel, policy=sample_from_measure_policy)
    num_integral, _ = quad(integrand, measure.domain[0], measure.domain[1])
    bq_integral, _ = bq.integrate(fun=f1d, measure=measure, nevals=250)
    np.testing.assert_almost_equal(bq_integral.mean, num_integral, decimal=2)


def test_bmc_without_rng_raises_error():
    """BMC policy requires a random number generator to be passed."""
    with pytest.raises(ValueError):
        dummy_input_dim = (
            1  # irrelevant for the test below, but required from the interface
        )
        BayesianQuadrature.from_interface(
            input_dim=dummy_input_dim, policy="bmc", rng=None
        )
