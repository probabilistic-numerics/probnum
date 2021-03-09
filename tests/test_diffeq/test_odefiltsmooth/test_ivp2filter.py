"""While previously, this module contained the tests for functions in the
`diffeq.odefiltsmooth.ivp2filter` module, since this module has become obsolete, we test
its replacement (`GaussianIVPFilter.string_to_measurement_model`) here.

With the next refactoring of `probsolve_ivp` and its tests, please
refactor this module, too.
"""
import unittest

import numpy as np

import probnum.diffeq as pnd
import probnum.filtsmooth as pnfs
import probnum.random_variables as pnrv
from tests.testing import NumpyAssertions


class Ivp2FilterTestCase(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        """We need a Prior object and an IVP object (with derivatives) to run the
        tests."""
        y0 = pnrv.Constant(np.array([20.0, 15.0]))
        self.ivp = pnd.lotkavolterra([0.4124, 1.15124], y0)
        self.prior = pnfs.statespace.IBM(ordint=2, spatialdim=2)
        self.evlvar = 0.0005123121


class TestIvp2Ekf0(Ivp2FilterTestCase):
    """Do ivp2ekf0, ivp2ekf1 return the right objects?

    Do the measurement models work? Do the initial values work?
    """

    def test_ivp2ekf0_output(self):
        measmod = pnd.GaussianIVPFilter.string_to_measurement_model(
            "ekf0", self.ivp, self.prior, self.evlvar
        )
        self.assertIsInstance(measmod, pnfs.DiscreteEKFComponent)

    def test_ekf0_measmod(self):
        measmod = pnd.GaussianIVPFilter.string_to_measurement_model(
            "eks0", self.ivp, self.prior, self.evlvar
        )
        random_time, random_eval = np.random.rand(), np.random.rand(
            self.prior.dimension
        )
        e0, e1 = self.prior.proj2coord(0), self.prior.proj2coord(1)
        expected = e1 @ random_eval - self.ivp.rhs(random_time, e0 @ random_eval)
        received, _ = measmod.forward_realization(random_eval, random_time)

        self.assertAllClose(expected, received.mean)


class TestIvp2Ekf1(Ivp2FilterTestCase):
    """Do ivp2ekf0, ivp2ekf1 return the right objects?

    Do the measurement models work? Do the initial values work?
    """

    def test_ivp2ekf1_output(self):
        measmod = pnd.GaussianIVPFilter.string_to_measurement_model(
            "ekf1", self.ivp, self.prior, self.evlvar
        )
        self.assertIsInstance(measmod, pnfs.DiscreteEKFComponent)

    def test_ekf1_measmod(self):
        measmod = pnd.GaussianIVPFilter.string_to_measurement_model(
            "eks1", self.ivp, self.prior, self.evlvar
        )
        random_time, random_eval = np.random.rand(), np.random.rand(
            self.prior.dimension
        )
        e0, e1 = self.prior.proj2coord(0), self.prior.proj2coord(1)
        expected = e1 @ random_eval - self.ivp.rhs(random_time, e0 @ random_eval)
        received, _ = measmod.forward_realization(random_eval, random_time)

        self.assertAllClose(expected, received.mean)


class TestIvpUkf(Ivp2FilterTestCase):
    """Do ivp2ekf0, ivp2ekf1 return the right objects?

    Do the measurement models work? Do the initial values work?
    """

    def test_ivp2ukf_output(self):
        measmod = pnd.GaussianIVPFilter.string_to_measurement_model(
            "ukf", self.ivp, self.prior, self.evlvar
        )
        self.assertIsInstance(measmod, pnfs.DiscreteUKFComponent)

    def test_ukf_measmod(self):
        measmod = pnd.GaussianIVPFilter.string_to_measurement_model(
            "uks", self.ivp, self.prior, self.evlvar
        )
        random_time, random_eval = np.random.rand(), np.random.rand(
            self.prior.dimension
        )
        e0, e1 = self.prior.proj2coord(0), self.prior.proj2coord(1)
        expected = e1 @ random_eval - self.ivp.rhs(random_time, e0 @ random_eval)
        received, _ = measmod.forward_realization(random_eval, random_time)

        self.assertAllClose(expected, received.mean)
