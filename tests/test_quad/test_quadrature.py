"""
We set up an MC quadrature rule as a test case.
"""
import unittest

import numpy as np

from probnum.quad.polynomial import polynomialquadrature


class TestQuadrature(unittest.TestCase):
    """
    Monte Carlo quadrature on 10000 points as a test case.
    """

    def setUp(self):
        npts = 10000
        ndim = 1
        nodes = np.random.rand(npts, ndim)
        weights = np.ones(npts) / npts
        bounds = np.array([[0.0, 1.0]])
        self.quad = polynomialquadrature.PolynomialQuadrature(nodes, weights, bounds)

    def test_compute(self):
        def testfct(x):
            return 10 * x ** 3 - x  # true integral: 2.5*x**4 - 0.5*x**2 + const

        res_seq = self.quad.integrate(testfct, isvectorized=False)
        res_vec = self.quad.integrate(testfct, isvectorized=True)
        self.assertLess(np.abs(res_seq - 2.0), 0.1)
        self.assertLess(np.abs(res_vec - 2.0), 0.1)
        self.assertLess(np.abs(res_vec - res_seq), 1e-10)

    def test_wrong_inputs(self):
        npts = 10
        ndim = 1
        good_nodes = np.random.rand(npts, ndim)
        bad_nodes = np.random.rand(npts)
        good_weights = np.ones(npts) / npts
        bad_weights = np.ones((npts, 1)) / npts
        incomp_weights = np.ones(npts + 1) / npts
        good_ilbds = np.array([[0.0, 1.0]])
        bad_ilbds = np.array([0.0, 1.0])
        with self.assertRaises(ValueError):
            polynomialquadrature.PolynomialQuadrature(
                good_nodes, good_weights, bad_ilbds
            )
        with self.assertRaises(ValueError):
            polynomialquadrature.PolynomialQuadrature(
                good_nodes, bad_weights, good_ilbds
            )
        with self.assertRaises(ValueError):
            polynomialquadrature.PolynomialQuadrature(
                good_nodes, incomp_weights, good_ilbds
            )
        with self.assertRaises(ValueError):
            polynomialquadrature.PolynomialQuadrature(
                bad_nodes, good_weights, good_ilbds
            )
