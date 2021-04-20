"""Test cases for Bayesian quadrature."""

from functools import partial

import numpy as np
import pytest
from scipy.integrate import quad

from probnum.quad import BayesianQuadrature, bayesquad
from probnum.quad.bq_methods.belief_updates import BQStandardBeliefUpdate
from probnum.quad.bq_methods.stop_criteria import MaxNevals
from probnum.quad.policies import sample_from_measure
from probnum.randvars import Normal


@pytest.mark.parametrize("input_dim", [1], ids=["dim1"])
def test_type_1d(f1d, kernel, measure, input_dim):
    """Test that BQ outputs normal random variables for 1D integrands."""
    integral, _ = bayesquad(
        fun=f1d, input_dim=input_dim, kernel=kernel, measure=measure, max_nevals=10
    )
    assert isinstance(integral, Normal)


@pytest.mark.parametrize("input_dim", [1])
def test_integral_values_1d(f1d, kernel, measure, input_dim):
    """Test numerically that BQ computes 1D integrals correctly."""

    # numerical integral
    # pylint: disable=invalid-name
    def integrand(x):
        return f1d(x) * measure(x)

    # pylint: disable=invalid-name
    bq_integral, _ = bayesquad(
        fun=f1d, input_dim=input_dim, kernel=kernel, measure=measure, max_nevals=250
    )
    num_integral, _ = quad(integrand, measure.domain[0], measure.domain[1])
    np.testing.assert_almost_equal(bq_integral.mean, num_integral, decimal=2)
