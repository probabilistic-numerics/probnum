"""
Test cases for Gaussian Filtering and Smoothing
"""
import unittest

import numpy as np

from probnum.filtsmooth import (
    DiscreteGaussianLTIModel,
    generate_dd,
    LTISDEModel,
    generate_cd,
    DiscreteGaussianModel,
)
from probnum.random_variables import Normal
from tests.testing import NumpyAssertions

__all__ = [
    "CarTrackingDDTestCase",
    "OrnsteinUhlenbeckCDTestCase",
    "PendulumNonlinearDDTestCase",
]


class CarTrackingDDTestCase(unittest.TestCase, NumpyAssertions):
    """
    Car tracking: Ex. 4.3 in Bayesian Filtering and Smoothing
    """

    delta_t = 0.2
    var = 0.5
    dynamat = np.eye(4) + delta_t * np.diag(np.ones(2), 2)
    dynadiff = (
        np.diag(np.array([delta_t ** 3 / 3, delta_t ** 3 / 3, delta_t, delta_t]))
        + np.diag(np.array([delta_t ** 2 / 2, delta_t ** 2 / 2]), 2)
        + np.diag(np.array([delta_t ** 2 / 2, delta_t ** 2 / 2]), -2)
    )
    measmat = np.eye(2, 4)
    measdiff = var * np.eye(2)
    mean = np.zeros(4)
    cov = 0.5 * var * np.eye(4)

    def setup_cartracking(self):
        self.dynmod = DiscreteGaussianLTIModel(
            dynamat=self.dynamat, forcevec=np.zeros(4), diffmat=self.dynadiff
        )
        self.measmod = DiscreteGaussianLTIModel(
            dynamat=self.measmat, forcevec=np.zeros(2), diffmat=self.measdiff
        )
        self.initrv = Normal(self.mean, self.cov)
        self.tms = np.arange(0, 20, self.delta_t)
        self.states, self.obs = generate_dd(
            self.dynmod, self.measmod, self.initrv, self.tms
        )


class OrnsteinUhlenbeckCDTestCase(unittest.TestCase, NumpyAssertions):
    """
    Ornstein Uhlenbeck process as a test case.
    """

    delta_t = 0.2
    lam, q, r = 0.21, 0.5, 0.1
    drift = -lam * np.eye(1)
    force = np.zeros(1)
    disp = np.eye(1)
    diff = q * np.eye(1)

    def setup_ornsteinuhlenbeck(self):
        self.dynmod = LTISDEModel(
            driftmatrix=self.drift,
            force=self.force,
            dispmatrix=self.disp,
            diffmatrix=self.diff,
        )
        self.measmod = DiscreteGaussianLTIModel(
            dynamat=np.eye(1), forcevec=np.zeros(1), diffmat=self.r * np.eye(1)
        )
        self.initrv = Normal(10 * np.ones(1), np.eye(1))
        self.tms = np.arange(0, 20, self.delta_t)
        self.states, self.obs = generate_cd(
            dynmod=self.dynmod, measmod=self.measmod, initrv=self.initrv, times=self.tms
        )


class PendulumNonlinearDDTestCase(unittest.TestCase, NumpyAssertions):
    def setup_pendulum(self):
        delta_t = 0.0075
        var = 0.32 ** 2
        g = 9.81

        def f(t, x):
            x1, x2 = x
            y1 = x1 + x2 * delta_t
            y2 = x2 - g * np.sin(x1) * delta_t
            return np.array([y1, y2])

        def df(t, x):
            x1, x2 = x
            y1 = [1, delta_t]
            y2 = [-g * np.cos(x1) * delta_t, 1]
            return np.array([y1, y2])

        def h(t, x):
            x1, x2 = x
            return np.array([np.sin(x1)])

        def dh(t, x):
            x1, x2 = x
            return np.array([[np.cos(x1), 0.0]])

        q = 1.0 * (
            np.diag(np.array([delta_t ** 3 / 3, delta_t]))
            + np.diag(np.array([delta_t ** 2 / 2]), 1)
            + np.diag(np.array([delta_t ** 2 / 2]), -1)
        )
        self.r = var * np.eye(1)
        initmean = np.ones(2)
        initcov = var * np.eye(2)
        self.dynamod = DiscreteGaussianModel(f, lambda t: q, df)
        self.measmod = DiscreteGaussianModel(h, lambda t: self.r, dh)
        self.initrv = Normal(initmean, initcov)
        self.tms = np.arange(0, 4, delta_t)
        self.q = q
        self.states, self.obs = generate_dd(
            self.dynamod, self.measmod, self.initrv, self.tms
        )
