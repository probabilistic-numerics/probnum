import unittest

import numpy as np

from probnum.utils import arrayutils


class TestAssertIsArray(unittest.TestCase):
    def test_assert_is_1d_ndarray_pass(self):
        arr = np.random.rand(4)
        arrayutils.assert_is_1d_ndarray(arr)
        self.assertEqual(1, 1)

    def test_assert_is_1d_ndarray_fail(self):
        arr_wrong = np.random.rand(4, 1)
        with self.assertRaises(ValueError):
            arrayutils.assert_is_1d_ndarray(arr_wrong)

        float_wrong = np.random.rand()
        with self.assertRaises(ValueError):
            arrayutils.assert_is_1d_ndarray(float_wrong)

    def test_assert_is_2d_ndarray_pass(self):
        arr = np.random.rand(4, 1)
        arrayutils.assert_is_2d_ndarray(arr)
        self.assertEqual(1, 1)

    def test_assert_is_2d_ndarray_fail(self):
        arr_wrong1 = np.random.rand(4)
        with self.assertRaises(ValueError):
            arrayutils.assert_is_2d_ndarray(arr_wrong1)

        arr_wrong2 = np.random.rand(4, 3, 2)
        with self.assertRaises(ValueError):
            arrayutils.assert_is_2d_ndarray(arr_wrong2)

        float_wrong = np.random.rand()
        with self.assertRaises(ValueError):
            arrayutils.assert_is_2d_ndarray(float_wrong)


def _set_in_bounds(ptset, ilbds):
    for (idx, col) in enumerate(ptset.T):
        if np.amin(col) != ilbds[idx, 0]:
            return False
        if np.amax(col) != ilbds[idx, 1]:
            return False
    return True
