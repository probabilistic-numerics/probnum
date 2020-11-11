import unittest

import numpy as np

import probnum.random_variables as pnrv
from probnum.filtsmooth.gaussfiltsmooth import IteratedKalman, Kalman, StoppingCriterion, KalmanPosterior

from .filtsmooth_testcases import OrnsteinUhlenbeckCDTestCase


class MockStoppingCriterion(StoppingCriterion):
    """Mock object that does 5 iterations of each predict, update and filtsmooth."""

    def __init__(self, max_num_updates=5):
        super().__init__(
            max_num_predicts_per_step=max_num_updates,
            max_num_updates_per_step=max_num_updates,
            max_num_filtsmooth_iterations=max_num_updates,
        )

    def continue_predict_iteration(self, pred_rv=None, info_pred=None):
        """Do we continue iterating the update step of the filter?"""
        if self.num_predict_iterations >= self.max_num_predicts_per_step:
            return False
        self.num_predict_iterations += 1
        return True

    def continue_filtsmooth_iteration(self, kalman_posterior=None):
        """If implemented, iterated_filtsmooth() is unlocked."""
        if self.num_filtsmooth_iterations >= self.max_num_filtsmooth_iterations:
            return False
        self.num_filtsmooth_iterations += 1
        return True

    def continue_update_iteration(self, upd_rv=None, meas_rv=None, info_upd=None):
        """
        When do we stop iterating the filter steps. Default is true.
        If, e.g. IEKF is wanted, overwrite with something that does not always return True.
        """
        if self.num_update_iterations >= self.max_num_updates_per_step:
            return False
        self.num_update_iterations += 1
        return True


class MockKalmanPosterior(KalmanPosterior):
    def __init__(self):
        self.times = np.arange(0.0, 1.1, 0.1)
        self.means = np.random.rand(10, 1)
        self.covs = np.random.rand(10, 1, 1)

    def __call_(self, t):
        if t in self.times:
            idx = (self.times <= t).sum() - 1
            return pnrv.Normal(self.means[idx], self.covs[idx])
        raise RuntimeError("beyond here is irrelevant for the test!")


class TestIteratedKalman(OrnsteinUhlenbeckCDTestCase):
    def setUp(self):
        super().setup_ornsteinuhlenbeck()
        kalman = Kalman(self.dynmod, self.measmod, self.initrv)
        self.max_updates = 5
        self.method = IteratedKalman(
            kalman,
            stoppingcriterion=MockStoppingCriterion(max_num_updates=self.max_updates),
        )

    def test_filter_step(self):
        self.assertEqual(self.method.stoppingcriterion.num_predict_iterations, 0)
        self.assertEqual(self.method.stoppingcriterion.num_update_iterations, 0)
        self.method.filter_step(start=0.0, stop=1.0, current_rv=self.initrv, data=0.0)
        self.assertEqual(self.method.stoppingcriterion.num_predict_iterations, 5)
        self.assertEqual(self.method.stoppingcriterion.num_update_iterations, 5)

    def test_iterated_filtsmooth(self):
        self.assertEqual(self.method.stoppingcriterion.num_filtsmooth_iterations, 0)
        self.method.iterated_filtsmooth(self.obs, self.tms)
        self.assertEqual(self.method.stoppingcriterion.num_filtsmooth_iterations, 5)


if __name__ == "__main__":
    unittest.main()
