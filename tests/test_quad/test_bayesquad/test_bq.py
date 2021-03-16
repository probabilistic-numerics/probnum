"""Test cases for Bayesian quadrature."""

import numpy as np

from probnum.kernels import ExpQuad
from probnum.quad import BayesianQuadrature, GaussianMeasure
from probnum.random_variables import Normal


def f1d(x):
    return x ** 2


dim = 1
k = ExpQuad(dim)
measure = GaussianMeasure(0.0, 1.0)
bq = BayesianQuadrature(f1d, k, measure)


def test_type():
    F, _ = bq.integrate(10)
    assert isinstance(F, Normal)
