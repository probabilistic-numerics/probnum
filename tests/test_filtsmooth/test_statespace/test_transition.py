import unittest

import numpy as np

import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv

TEST_NDIM = 10


class MockTransition(pnfss.Transition):
    """Empty transition object to test generate() function."""

    def transition_realization(self, real, start, stop=None, **kwargs):
        return pnrv.Dirac(real), {}

    def transition_rv(self, rv, start, stop=None, **kwargs):
        return rv, {}


class TestGenerate(unittest.TestCase):
    def test_generate(self):
        mocktrans = MockTransition()
        initrv = pnrv.Dirac(np.random.rand(TEST_NDIM))
        times = np.arange(0.0, 13.0, 1.0)  # length 13
        states, obs = pnfss.generate(mocktrans, mocktrans, initrv, times)
        self.assertEqual(states.shape[0], len(times))
        self.assertEqual(states.shape[1], TEST_NDIM)
        self.assertEqual(obs.shape[0], len(times) - 1)
        self.assertEqual(obs.shape[1], TEST_NDIM)


if __name__ == "__main__":
    unittest.main()
