import unittest

import numpy as np

import probnum.filtsmooth as pnfs
from tests.testing import NumpyAssertions

from .filtsmooth_testcases import PendulumNonlinearDDTestCase

np.random.seed(5472)


class TestExtendedKalmanPendulum(
    PendulumNonlinearDDTestCase, unittest.TestCase, NumpyAssertions
):
    """
    We test on the pendulum example 5.1 in BFaS.
    """

    visualise = False  # show plots or not?

    def setUp(self):
        super().setup_pendulum()
        self.ekf_meas = pnfs.DiscreteEKF(self.measmod)
        self.ekf_dyna = pnfs.DiscreteEKF(self.dynamod)
        self.method = pnfs.Kalman(self.ekf_dyna, self.ekf_meas, self.initrv)
