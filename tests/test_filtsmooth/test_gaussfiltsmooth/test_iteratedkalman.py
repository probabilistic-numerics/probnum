import unittest
import numpy as np

from probnum.filtsmooth.gaussfiltsmooth import IteratedKalman, Kalman, StoppingCriterion

from .filtsmooth_testcases import OrnsteinUhlenbeckCDTestCase


class MockStoppingCriterion(StoppingCriterion):
    """Mock object that does 5 iterations of each filter and filtsmooth."""

    def __init__(self, max_num_updates=5):
        self.num_filter_updates = 0
        # self.num_filtsmooth_updates = 0
        self.max_num_updates = max_num_updates

    def continue_filter_updates(
        self,
        predrv=None,
        info_pred=None,
        filtrv=None,
        meas_rv=None,
        info_upd=None,
        **kwargs
    ):
        """
        When do we stop iterating the filter steps. Default is true.
        If, e.g. IEKF is wanted, overwrite with something that does not always return True.
        """
        if self.num_filter_updates < self.max_num_updates:
            self.num_filter_updates += 1
            return True
        else:
            return False

    def continue_filtsmooth_updates(self, **kwargs):
        """If implemented, iterated_filtsmooth() is unlocked."""
        if self.num_filtsmooth_updates < self.max_num_updates:
            self.num_filtsmooth_updates += 1
            return True
        else:
            return False


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
        self.assertEqual(self.method.stoppingcriterion.num_filter_updates, 0)
        self.method.filter_step(start=0.0, stop=1.0, current_rv=self.initrv, data=0.0)
        self.assertEqual(self.method.stoppingcriterion.num_filter_updates, 5)

    # def test_filtsmooth_step(self):
    #     self.assertEqual(self.method.stoppingcriterion.num_filter_updates, 0)
    #     self.method.iterated_filtsmooth(current_rv=self.initrv, data=0.)
    #     self.assertEqual(self.method.stoppingcriterion.num_filter_updates, 5)


if __name__ == '__main__':
    unittest.main()
