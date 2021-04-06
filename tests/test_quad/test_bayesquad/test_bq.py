"""Test cases for Bayesian quadrature."""

import numpy as np
import pytest
from scipy.integrate import quad

from probnum.quad import BayesianQuadrature, sample_from_measure
from probnum.randvars import Normal


@pytest.mark.parametrize("input_dim", [1], ids=["dim1"])
def test_type_1d(f1d, kernel, measure):
    """Test that BQ outputs normal random variables for 1D integrands."""
    # pylint: disable=invalid-name
    bq = BayesianQuadrature(kernel=kernel, policy=sample_from_measure)
    integral, _ = bq.integrate(fun=f1d, measure=measure, nevals=10)
    assert isinstance(integral, Normal)


@pytest.mark.parametrize("input_dim", [1])
def test_integral_values_1d(f1d, kernel, measure):
    """Test numerically that BQ computes 1D integrals correctly."""

    # numerical integral
    # pylint: disable=invalid-name
    def integrand(x):
        return f1d(x) * measure(x)

    # pylint: disable=invalid-name
    bq = BayesianQuadrature(kernel=kernel, policy=sample_from_measure)
    num_integral, _ = quad(integrand, measure.domain[0], measure.domain[1])
    bq_integral, _ = bq.integrate(fun=f1d, measure=measure, nevals=250)
    np.testing.assert_almost_equal(bq_integral.mean, num_integral, decimal=2)
