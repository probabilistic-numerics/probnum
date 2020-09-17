import unittest

import numpy as np

from probnum.utils import fctutils


class TestAssertsEvaluatesToScalar(unittest.TestCase):
    """Test case for utility functions dealing with functions."""

    def test_assert_evaluates_to_scalar_pass(self):
        def fct(x):
            return np.linalg.norm(x)

        inp = np.random.rand(10)
        fctutils.assert_evaluates_to_scalar(fct, inp)
        self.assertEqual(1, 1)

    def test_assert_evaluates_to_scalar_fail(self):
        def fct(x):
            return np.array(x)

        inp = np.random.rand(10)
        with self.assertRaises(ValueError):
            fctutils.assert_evaluates_to_scalar(fct, inp)


def _set_in_bounds(ptset, ilbds):

    for (idx, col) in enumerate(ptset.T):
        if np.amin(col) != ilbds[idx, 0]:
            return False
        if np.amax(col) != ilbds[idx, 1]:
            return False
    return True
