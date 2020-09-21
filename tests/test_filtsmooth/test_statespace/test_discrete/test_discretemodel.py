import unittest
import numpy as np
import probnum.random_variables as rvs
from probnum.filtsmooth.statespace.discrete import discretemodel

TEST_NDIM = 2


class MockDiscreteModel(discretemodel.DiscreteModel):
    def sample(self, time, state, **kwargs):
        return state

    def transition_rv(self, rv, *args):
        return rv

    def transition_array(self, arr, *args):
        return rvs.asrandvar(arr)

    @property
    def dimension(self):
        return TEST_NDIM


class TestDiscreteModel(unittest.TestCase):
    def setUp(self):
        self.mdm = MockDiscreteModel()

    def test_call_rv(self):
        out = self.mdm(rvs.Dirac(0.1))
        self.assertIsInstance(out, rvs.RandomVariable)

    def test_call_arr(self):
        out = self.mdm(np.random.rand(4))
        self.assertIsInstance(out, rvs.RandomVariable)

    def test_dimension(self):
        self.assertEqual(self.mdm.dimension, TEST_NDIM)
