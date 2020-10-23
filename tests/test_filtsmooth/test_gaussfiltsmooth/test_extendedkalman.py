import unittest

import probnum.filtsmooth as pnfs

from . import filtsmooth_testcases as cases


class TestContinuousEKFComponent(unittest.TestCase):
    """Implementation incomplete, hence check that an error is raised."""

    def test_notimplementederror(self):
        sde = pnfs.statespace.SDE(None, None, None)  # content is irrelevant.
        with self.assertRaises(NotImplementedError):
            pnfs.ContinuousEKFComponent(sde)


class TestDiscreteEKFComponent(unittest.TestCase):
    def setUp(self):
        self.nonlinear_model, _, self.initrv, _ = cases.pendulum()
        self.linearised_model = pnfs.DiscreteEKFComponent(self.nonlinear_model)

    def test_transition_rv(self):
        """transition_rv() not possible for original model but for the linearised model"""

        with self.subTest("Baseline should not work."):
            with self.assertRaises(NotImplementedError):
                self.nonlinear_model.transition_rv(self.initrv, 0.0)
        with self.subTest("Linearisation happens."):
            self.linearised_model.transition_rv(self.initrv, 0.0)


class TestPendulumEKF(cases.PendulumNonlinearDDTestCase, unittest.TestCase):

    visualise = False

    def setUp(self):

        super().setup_pendulum()
        self.ekf_meas = pnfs.DiscreteEKFComponent(self.measmod)
        self.ekf_dyna = pnfs.DiscreteEKFComponent(self.dynamod)
        self.method = pnfs.Kalman(self.ekf_dyna, self.ekf_meas, self.initrv)


#
#
#
#
#
#
#
#
#
#
#
#
# class TestExtendedKalmanPendulum(
#     PendulumNonlinearDDTestCase, unittest.TestCase, NumpyAssertions
# ):
#     """
#     We test on the pendulum example 5.1 in BFaS.
#     """
#
#     visualise = False  # show plots or not?
#
#     def setUp(self):
#         super().setup_pendulum()
#         self.ekf_meas = pnfs.DiscreteEKF(self.measmod)
#         self.ekf_dyna = pnfs.DiscreteEKF(self.dynamod)
#         self.method = pnfs.Kalman(self.ekf_dyna, self.ekf_meas, self.initrv)
