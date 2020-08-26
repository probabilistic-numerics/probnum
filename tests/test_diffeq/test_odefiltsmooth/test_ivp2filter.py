"""
Checks for ivp2filter functions.

Covers:
    * Are the output ExtendedKalman/UnscentedKalman objects?
    * Does the measurement model do what we think it does
    * Are the initial values initialised truthfully (y0, f(y0), Jf(y0)f(y0), ...)
"""
import unittest

import numpy as np

from probnum.diffeq import IBM, ivp2filter, lotkavolterra
from probnum.filtsmooth import ExtendedKalman, UnscentedKalman
from probnum.random_variables import Dirac
from tests.testing import NumpyAssertions


class Ivp2FilterTestCase(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        """We need a Prior object and an IVP object (with derivatives) to run the tests."""
        y0 = Dirac(np.array([20.0, 15.0]))
        self.ivp = lotkavolterra([0.4124, 1.15124], y0)
        self.prior = IBM(ordint=2, spatialdim=2, diffconst=1.7685)
        self.evlvar = 0.0005123121


class TestIvp2Ekf0(Ivp2FilterTestCase):
    """
    Do ivp2ekf0, ivp2ekf1 return the right objects?
    Do the measurement models work? Do the initial values work?
    """

    def test_ivp2ekf0_output(self):
        filtsmooth_object = ivp2filter.ivp2ekf0(self.ivp, self.prior, self.evlvar)
        self.assertEqual(issubclass(type(filtsmooth_object), ExtendedKalman), True)

    def test_ekf0_measmod(self):
        filtsmooth_object = ivp2filter.ivp2ekf0(self.ivp, self.prior, self.evlvar)
        random_time, random_eval = np.random.rand(), np.random.rand(self.prior.ndim)
        e0, e1 = self.prior.proj2coord(0), self.prior.proj2coord(1)
        expected_measmodel_output = e1 @ random_eval - self.ivp.rhs(
            random_time, e0 @ random_eval
        )
        measmodel_output = filtsmooth_object.measurementmodel.dynamics(
            random_time, random_eval
        )
        self.assertAllClose(expected_measmodel_output, measmodel_output)

    def test_ekf0_initialdistribution(self):
        filtsmooth_object = ivp2filter.ivp2ekf0(self.ivp, self.prior, self.evlvar)
        expected_initval = np.array(
            [
                self.ivp.initrv.mean,
                self.ivp(self.ivp.t0, self.ivp.initrv.mean),
                self.ivp.jacobian(self.ivp.t0, self.ivp.initrv.mean)
                @ self.ivp(self.ivp.t0, self.ivp.initrv.mean),
            ]
        )
        self.assertAllClose(
            filtsmooth_object.initialrandomvariable.mean, expected_initval.T.flatten()
        )


class TestIvp2Ekf1(Ivp2FilterTestCase):
    """
    Do ivp2ekf0, ivp2ekf1 return the right objects?
    Do the measurement models work? Do the initial values work?
    """

    def test_ivp2ekf1_output(self):
        filtsmooth_object = ivp2filter.ivp2ekf1(self.ivp, self.prior, self.evlvar)
        self.assertEqual(issubclass(type(filtsmooth_object), ExtendedKalman), True)

    def test_ekf1_measmod(self):
        filtsmooth_object = ivp2filter.ivp2ekf1(self.ivp, self.prior, self.evlvar)
        random_time, random_eval = np.random.rand(), np.random.rand(self.prior.ndim)
        e0, e1 = self.prior.proj2coord(0), self.prior.proj2coord(1)
        expected_measmodel_output = e1 @ random_eval - self.ivp.rhs(
            random_time, e0 @ random_eval
        )
        measmodel_output = filtsmooth_object.measurementmodel.dynamics(
            random_time, random_eval
        )
        self.assertAllClose(expected_measmodel_output, measmodel_output)

    def test_ekf1_initialdistribution(self):
        filtsmooth_object = ivp2filter.ivp2ekf1(self.ivp, self.prior, self.evlvar)
        expected_initval = np.array(
            [
                self.ivp.initrv.mean,
                self.ivp(self.ivp.t0, self.ivp.initrv.mean),
                self.ivp.jacobian(self.ivp.t0, self.ivp.initrv.mean)
                @ self.ivp(self.ivp.t0, self.ivp.initrv.mean),
            ]
        )
        self.assertAllClose(
            filtsmooth_object.initialrandomvariable.mean, expected_initval.T.flatten()
        )


class TestIvpUkf(Ivp2FilterTestCase):
    """
    Do ivp2ekf0, ivp2ekf1 return the right objects?
    Do the measurement models work? Do the initial values work?
    """

    def test_ivp2ukf_output(self):
        filtsmooth_object = ivp2filter.ivp2ukf(self.ivp, self.prior, self.evlvar)
        self.assertEqual(issubclass(type(filtsmooth_object), UnscentedKalman), True)

    def test_ukf_measmod(self):
        filtsmooth_object = ivp2filter.ivp2ukf(self.ivp, self.prior, self.evlvar)
        random_time, random_eval = np.random.rand(), np.random.rand(self.prior.ndim)
        e0, e1 = self.prior.proj2coord(0), self.prior.proj2coord(1)
        expected_measmodel_output = e1 @ random_eval - self.ivp.rhs(
            random_time, e0 @ random_eval
        )
        measmodel_output = filtsmooth_object.measurementmodel.dynamics(
            random_time, random_eval
        )
        self.assertAllClose(expected_measmodel_output, measmodel_output)

    def test_ukf_initialdistribution(self):
        filtsmooth_object = ivp2filter.ivp2ukf(self.ivp, self.prior, self.evlvar)
        expected_initval = np.array(
            [
                self.ivp.initrv.mean,
                self.ivp(self.ivp.t0, self.ivp.initrv.mean),
                self.ivp.jacobian(self.ivp.t0, self.ivp.initrv.mean)
                @ self.ivp(self.ivp.t0, self.ivp.initrv.mean),
            ]
        )
        self.assertAllClose(
            filtsmooth_object.initialrandomvariable.mean, expected_initval.T.flatten()
        )
