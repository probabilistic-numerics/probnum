"""
Here, we mostly test for polynomial exactness of degree n+1,
where n is the number of quadrature nodes.

The 1d interface is covered by TestGaussian
The nd interface is covered by TestPolynomialExactness
"""

import unittest

import numpy as np
import numpy.polynomial.polynomial as npoly

from probnum.quad.polynomial import clenshawcurtis


class TestPolynomialExactness(unittest.TestCase):
    """
    Tests whether Clenshaw-Curtis rule is exact on polynomials
    of maximum degree npts+1.
    """

    def setUp(self):
        """
        Set up disorderly setting:
        few number of points in giant domain, so that there is no
        coincidental exactness.
        """
        npts = [1, 5, 9, 13, 17, 21]  # a few test cases
        self.ndim = 2
        lwbds = np.array([-7648.5, -7345.3])
        uppbds = np.array([4564.9, 4534.4])
        self.ilbds = np.vstack((lwbds, uppbds)).T
        self.orders = npts

    def test_integrate_polyn_good_degree(self):
        """
        Check quadrature errors for polynomials of max. degree N+1
        which should be zero. The integral values on the given domain are of
        magnitue 1e90 for n=13, so if this relative error is machine precision, it
        must be exact.
        """
        for number in self.orders:
            cc = clenshawcurtis.ClenshawCurtis(number, self.ndim, self.ilbds)
            random_poly = np.random.randint(1, 11, (number + 1, number + 1))
            integrated_poly = npoly.polyint(npoly.polyint(random_poly).T).T

            # pylint: disable=cell-var-from-loop
            def testpoly(val):
                return npoly.polyval2d(val[:, 0], val[:, 1], c=random_poly)

            truesol = (
                npoly.polyval2d(self.ilbds[0, 1], self.ilbds[1, 1], c=integrated_poly)
                - npoly.polyval2d(self.ilbds[0, 1], self.ilbds[1, 0], c=integrated_poly)
                - npoly.polyval2d(self.ilbds[0, 0], self.ilbds[1, 1], c=integrated_poly)
                + npoly.polyval2d(self.ilbds[0, 0], self.ilbds[1, 0], c=integrated_poly)
            )

            abserror = np.abs(cc.integrate(testpoly, isvectorized=True) - truesol)
            relerror = abserror / np.abs(truesol)
            self.assertLess(relerror, 1e-14)

    def test_integrate_polyn_bad_degree(self):
        """
        Check quadrature errors for polynomials of max. degree N+1
        which should be zero. The integral values on the given domain are of
        magnitue 1e90 for n=13, so if this relative error is machine precision, it
        must be exact.
        """
        for number in self.orders:
            cc = clenshawcurtis.ClenshawCurtis(number, self.ndim, self.ilbds)
            bad_configs = [
                (number + 1, number + 2),
                (number + 2, number + 1),
                (number + 2, number + 2),
            ]
            for config in bad_configs:
                random_poly = np.random.randint(1, 11, config)
                integrated_poly = npoly.polyint(npoly.polyint(random_poly).T).T

                # pylint: disable=cell-var-from-loop
                def testpoly(val):
                    return npoly.polyval2d(val[:, 0], val[:, 1], c=random_poly)

                truesol = (
                    npoly.polyval2d(
                        self.ilbds[0, 1], self.ilbds[1, 1], c=integrated_poly
                    )
                    - npoly.polyval2d(
                        self.ilbds[0, 1], self.ilbds[1, 0], c=integrated_poly
                    )
                    - npoly.polyval2d(
                        self.ilbds[0, 0], self.ilbds[1, 1], c=integrated_poly
                    )
                    + npoly.polyval2d(
                        self.ilbds[0, 0], self.ilbds[1, 0], c=integrated_poly
                    )
                )

                abserror = np.abs(cc.integrate(testpoly, isvectorized=True) - truesol)
                relerror = abserror / np.abs(truesol)
                self.assertLess(1e-10, relerror)


THREE_SIGMA = 0.997300203936740


class TestGaussian(unittest.TestCase):
    """
    3*stdev covers 99.73 percent of the prob. mass
    we use half of it. Hence
    we expect to cover 99.73/2 percent

    https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    """

    def setUp(self):
        """
        We set up a very steep Gaussian,
        attempting to create a disorderly setting
        which is hard to integrate numerically.
        """

        mean = 2.5678
        var = 0.01

        def gaussian(x):

            return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

        ilbds = np.array([[mean - 3 * np.sqrt(var), mean]])
        self.cc = clenshawcurtis.ClenshawCurtis(npts_per_dim=35, ndim=1, bounds=ilbds)
        self.gaussian = gaussian

    def test_integral(self):
        """
        Relative error of 1e-14 seems like the max of
        what we can expect, given that THREE_SIGMA
        has 14 decimals.
        """
        approx = self.cc.integrate(self.gaussian)
        relerror = np.abs(approx - THREE_SIGMA / 2) / (THREE_SIGMA / 2)
        self.assertLess(relerror, 1e-14)
