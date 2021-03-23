"""Test cases for Bayesian quadrature."""

import numpy as np
import pytest
from scipy.integrate import quad

from probnum.kernels import ExpQuad
from probnum.quad import BayesianQuadrature, get_kernel_embedding, sample_from_measure
from probnum.randvars import Normal

test_funs_1d = [lambda x: x, lambda x: x ** 2, lambda x: np.sin(x)]


@pytest.mark.parametrize("f1d", test_funs_1d, ids=["x", "x^2", "sinx"])
@pytest.mark.parametrize("input_dim", [1])
def test_type_1d(f1d, kernel, measure):
    """Test that BQ outputs normal random variables for 1D integrands."""
    bq = BayesianQuadrature(kernel=kernel, policy=sample_from_measure)
    F, _ = bq.integrate(fun=f1d, measure=measure, nevals=10)
    assert isinstance(F, Normal)


@pytest.mark.parametrize("f1d", test_funs_1d, ids=["x", "x^2", "sinx"])
@pytest.mark.parametrize("input_dim", [1])
def test_integral_values_1d(f1d, measure, kernel):
    """Test numerically that BQ computes 1D integrals correctly."""

    # numerical integral
    def integrand(x):
        return f1d(x) * measure(x)

    bq = BayesianQuadrature(kernel=kernel, policy=sample_from_measure)

    num_integral, _ = quad(integrand, measure.domain[0], measure.domain[1])
    bq_integral, _ = F, _ = bq.integrate(fun=f1d, measure=measure, nevals=250)

    np.testing.assert_almost_equal(bq_integral.mean, num_integral, decimal=2)
