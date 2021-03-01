"""Checks for ivp2filter functions.

Covers:
    * Are the output ExtendedKalman/UnscentedKalman objects?
    * Does the measurement model do what we think it does
    * Are the initial values initialised truthfully (y0, f(y0), Jf(y0)f(y0), ...)
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
        filtsmooth_object = pnd.ivp2ekf0(self.ivp, self.prior, self.evlvar)
        self.assertIsInstance(
            filtsmooth_object.measurement_model, pnfs.DiscreteEKFComponent
        )

    def test_ekf0_measmod(self):
        kalman = pnd.ivp2ekf0(self.ivp, self.prior, self.evlvar)
        random_time, random_eval = np.random.rand(), np.random.rand(
            self.prior.dimension
        )
        e0, e1 = self.prior.proj2coord(0), self.prior.proj2coord(1)
        expected = e1 @ random_eval - self.ivp.rhs(random_time, e0 @ random_eval)
        received, _ = kalman.measurement_model.forward_realization(
            random_eval, random_time
        )

        self.assertAllClose(expected, received.mean)

    def test_ekf0_initialdistribution(self):
        filtsmooth_object = pnd.ivp2ekf0(self.ivp, self.prior, self.evlvar)
        expected_initval = np.array(
            [
                self.ivp.initrv.mean,
                self.ivp(self.ivp.t0, self.ivp.initrv.mean),
                self.ivp.jacobian(self.ivp.t0, self.ivp.initrv.mean)
                @ self.ivp(self.ivp.t0, self.ivp.initrv.mean),
            ]
        )
        self.assertAllClose(filtsmooth_object.initrv.mean, expected_initval.T.flatten())


class TestIvp2Ekf1(Ivp2FilterTestCase):
    """Do ivp2ekf0, ivp2ekf1 return the right objects?

    Do the measurement models work? Do the initial values work?
    """

    def test_ivp2ekf1_output(self):
        filtsmooth_object = pnd.ivp2ekf1(self.ivp, self.prior, self.evlvar)
        self.assertIsInstance(
            filtsmooth_object.measurement_model, pnfs.DiscreteEKFComponent
        )

    def test_ekf1_measmod(self):
        kalman = pnd.ivp2ekf1(self.ivp, self.prior, self.evlvar)
        random_time, random_eval = np.random.rand(), np.random.rand(
            self.prior.dimension
        )
        e0, e1 = self.prior.proj2coord(0), self.prior.proj2coord(1)
        expected = e1 @ random_eval - self.ivp.rhs(random_time, e0 @ random_eval)
        received, _ = kalman.measurement_model.forward_realization(
            random_eval, random_time
        )

        self.assertAllClose(expected, received.mean)

    def test_ekf1_initialdistribution(self):
        filtsmooth_object = pnd.ivp2ekf1(self.ivp, self.prior, self.evlvar)
        expected_initval = np.array(
            [
                self.ivp.initrv.mean,
                self.ivp(self.ivp.t0, self.ivp.initrv.mean),
                self.ivp.jacobian(self.ivp.t0, self.ivp.initrv.mean)
                @ self.ivp(self.ivp.t0, self.ivp.initrv.mean),
            ]
        )
        self.assertAllClose(filtsmooth_object.initrv.mean, expected_initval.T.flatten())


class TestIvpUkf(Ivp2FilterTestCase):
    """Do ivp2ekf0, ivp2ekf1 return the right objects?

    Do the measurement models work? Do the initial values work?
    """

    def test_ivp2ukf_output(self):
        filtsmooth_object = pnd.ivp2ukf(self.ivp, self.prior, self.evlvar)
        self.assertIsInstance(
            filtsmooth_object.measurement_model, pnfs.DiscreteUKFComponent
        )

    def test_ukf_measmod(self):
        kalman = pnd.ivp2ukf(self.ivp, self.prior, self.evlvar)
        random_time, random_eval = np.random.rand(), np.random.rand(
            self.prior.dimension
        )
        e0, e1 = self.prior.proj2coord(0), self.prior.proj2coord(1)
        expected = e1 @ random_eval - self.ivp.rhs(random_time, e0 @ random_eval)
        received, _ = kalman.measurement_model.forward_realization(
            random_eval, random_time
        )

        self.assertAllClose(expected, received.mean)

    def test_ukf_initialdistribution(self):
        filtsmooth_object = pnd.ivp2ukf(self.ivp, self.prior, self.evlvar)
        expected_initval = np.array(
            [
                self.ivp.initrv.mean,
                self.ivp(self.ivp.t0, self.ivp.initrv.mean),
                self.ivp.jacobian(self.ivp.t0, self.ivp.initrv.mean)
                @ self.ivp(self.ivp.t0, self.ivp.initrv.mean),
            ]
        )
        self.assertAllClose(filtsmooth_object.initrv.mean, expected_initval.T.flatten())
