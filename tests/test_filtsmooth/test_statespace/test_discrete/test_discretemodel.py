import unittest

from probnum.filtsmooth.statespace.discrete import discretemodel

TEST_NDIM = 2


class MockDiscreteModel(discretemodel.DiscreteModel):
    def sample(self, time, state, **kwargs):
        return state

    @property
    def ndim(self):
        return TEST_NDIM


class TestDiscreteModel(unittest.TestCase):
    def setUp(self):
        self.mdm = MockDiscreteModel()

    def test_sample(self):
        self.mdm.sample(0.0, 0.0)

    def test_ndim(self):
        self.assertEqual(self.mdm.ndim, TEST_NDIM)

    def test_pdf(self):
        with self.assertRaises(NotImplementedError):
            self.mdm.pdf(0.0, 0.0, 0.0)
