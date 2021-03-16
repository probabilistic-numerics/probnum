"""Test cases for Bayesian quadrature."""

import numpy as np
import pytest
from scipy.integrate import quad

from probnum.kernels import ExpQuad
from probnum.quad import BayesianQuadrature, GaussianMeasure, sample_from_measure
from probnum.random_variables import Normal


@pytest.mark.parametrize("f1d", [lambda x: x, lambda x: x ** 2, lambda x: np.sin(x)])
def test_type(f1d):
    random_state = np.random.RandomState(seed=0)
    dim = 1
    k = ExpQuad(dim)
    measure = GaussianMeasure(mean=0.0, cov=1.0, random_state=random_state)
    bq = BayesianQuadrature(kernel=k, policy=sample_from_measure)

    F, _ = bq.integrate(f1d, measure, 10)
    assert isinstance(F, Normal)


@pytest.mark.parametrize("f1d", [lambda x: x, lambda x: x ** 2, lambda x: np.sin(x)])
def test_integral_values_1d(f1d):

    # numerical integral
    def integrand(x):
        return f1d(x) * measure(x)

    random_state = np.random.RandomState(seed=0)
    dim = 1
    k = ExpQuad(dim)
    measure = GaussianMeasure(mean=0.25, cov=1.25, random_state=random_state)
    bq = BayesianQuadrature(kernel=k, policy=sample_from_measure)

    # get reasonable integration box to avoid trouble with quad and np.inf
    mean = measure.mean
    std = np.sqrt(measure.cov)
    integration_box = (mean - 5 * std, mean + 5 * std)

    num_integral, _ = quad(integrand, integration_box[0], integration_box[1])

    bq_integral, _ = F, _ = bq.integrate(f1d, measure, 25)

    np.testing.assert_almost_equal(bq_integral.mean, num_integral, decimal=2)
