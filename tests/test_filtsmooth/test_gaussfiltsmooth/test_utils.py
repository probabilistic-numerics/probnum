import unittest

import numpy as np

import probnum.filtsmooth as pnfs
from tests.testing import NumpyAssertions

TEST_NDIM = 5


class TestCholeskyUpdate(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        self.S1 = np.random.rand(TEST_NDIM)
        self.X1 = self.S1 @ self.S1.T + np.eye(TEST_NDIM)
        self.S2 = np.random.rand(TEST_NDIM, TEST_NDIM)
        self.X2 = self.S2 @ self.S2.T + np.eye(TEST_NDIM)
        self.L1 = np.linalg.cholesky(self.X1)
        self.L2 = np.linalg.cholesky(self.X2)
        self.H = np.random.rand(TEST_NDIM, TEST_NDIM)

    def test_yields_cholesky_both(self):
        received = pnfs.cholesky_update(self.H @ self.L1, self.L2)
        expected = np.linalg.cholesky(self.H @ self.X1 @ self.H.T + self.X2)
        self.assertAllClose(received, expected)

    def test_yields_cholesky_sum(self):
        received = pnfs.cholesky_update(self.L1, self.L2)
        expected = np.linalg.cholesky(self.X1 + self.X2)
        self.assertAllClose(received, expected)

    def test_yields_cholesky_prod(self):
        received = pnfs.cholesky_update(self.H @ self.L1)
        expected = np.linalg.cholesky(self.H @ self.X1 @ self.H.T)
        self.assertAllClose(received, expected)


if __name__ == "__main__":
    unittest.main()
