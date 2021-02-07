import unittest

import numpy as np

import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv

TEST_NDIM = 10


class MockTransition(pnfss.Transition):
    """Empty transition object to test generate() function."""

    def __init__(self):
        super().__init__(input_dim=TEST_NDIM, output_dim=TEST_NDIM)

    def forward_realization(self, real, **kwargs):
        return pnrv.Constant(real), {}

    def forward_rv(self, rv, **kwargs):
        return rv, {}

    def backward_realization(self, real, rv_past, **kwargs):
        raise NotImplementedError

    def backward_rv(self, rv_futu, rv_past, **kwargs):
        raise NotImplementedError


class TestGenerate(unittest.TestCase):
    def test_generate(self):
        mocktrans = MockTransition()
        initrv = pnrv.Constant(np.random.rand(TEST_NDIM))
        times = np.arange(0.0, 13.0, 1.0)  # length 13
        states, obs = pnfss.generate(mocktrans, mocktrans, initrv, times)
        self.assertEqual(states.shape[0], len(times))
        self.assertEqual(states.shape[1], TEST_NDIM)
        self.assertEqual(obs.shape[0], len(times))
        self.assertEqual(obs.shape[1], TEST_NDIM)


if __name__ == "__main__":
    unittest.main()
