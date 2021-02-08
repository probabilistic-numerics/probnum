import numpy as np
import pytest

import probnum.random_variables as pnrv
from probnum.filtsmooth import statespace as pnfss


class MockTransition(pnfss.Transition):
    """Empty transition object to test the generate() function."""

    def __init__(self, dim):
        super().__init__(input_dim=dim, output_dim=dim)

    def forward_realization(self, real, **kwargs):
        return pnrv.Constant(real), {}

    def forward_rv(self, rv, **kwargs):
        return rv, {}

    def backward_realization(self, *args, **kwargs):
        raise NotImplementedError

    def backward_rv(self, *args, **kwargs):
        raise NotImplementedError


def times_expected():
    return np.arange(0.0, 13.0, 1.0)


def times_single_point():
    return np.array([0.0])


@pytest.mark.parametrize("times", [times_expected(), times_single_point()])
@pytest.mark.parametrize("test_ndim", [0, 1, 3])
def test_generate_shapes(times, test_ndim):
    mocktrans = MockTransition(dim=test_ndim)
    initrv = pnrv.Constant(np.random.rand(test_ndim))
    states, obs = pnfss.generate(mocktrans, mocktrans, initrv, times)

    assert states.shape[0] == len(times)
    assert states.shape[1] == test_ndim
    assert obs.shape[0] == len(times)
    assert obs.shape[1] == test_ndim


# import unittest
#
# import numpy as np
#
# import probnum.filtsmooth.statespace as pnfss
# import probnum.random_variables as pnrv
#
# TEST_NDIM = 10
#
#
#
#
# class TestGenerate(unittest.TestCase):
#     def test_generate(self):
#         mocktrans = MockTransition()
#         initrv = pnrv.Constant(np.random.rand(TEST_NDIM))
#         times = np.arange(0.0, 13.0, 1.0)  # length 13
#         states, obs = pnfss.generate(mocktrans, mocktrans, initrv, times)
#         self.assertEqual(states.shape[0], len(times))
#         self.assertEqual(states.shape[1], TEST_NDIM)
#         self.assertEqual(obs.shape[0], len(times))
#         self.assertEqual(obs.shape[1], TEST_NDIM)
#
#
# if __name__ == "__main__":
#     unittest.main()
