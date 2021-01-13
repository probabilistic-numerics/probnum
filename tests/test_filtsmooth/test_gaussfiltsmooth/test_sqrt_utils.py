import numpy as np
import pytest

import probnum.filtsmooth as pnfs
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions


@pytest.fixture
def S1(d):
    return random_spd_matrix(dim=d)


@pytest.fixture
def spdmat2(d):
    return random_spd_matrix(dim=d)


spdmat = random_spd_matrix(3)
randmat = np.random.rand(3, 3)


@pytest.mark.parametrize("S1", [spdmat, randmat @ spdmat])
@pytest.mark.parametrize("S2", [spdmat, randmat @ spdmat])
def test_cholesky_update_sum(S1, S2):
    S3 = pnfs.cholesky_update(S1, S2)
    np.testing.assert_allclose(S3 @ S3.T, S1 @ S1.T + S2 @ S2.T)
    np.testing.assert_allclose(np.tril(S3), S3)
    np.testing.assert_allclose(np.diag(S3), np.abs(np.diag(S3)))


#
#
#
#
#
#
#
#
#
# class TestCholeskyUpdate(unittest.TestCase, NumpyAssertions):
#     def setUp(self):
#         self.S1 = np.random.rand(TEST_NDIM)
#         self.X1 = self.S1 @ self.S1.T + np.eye(TEST_NDIM)
#         self.S2 = np.random.rand(TEST_NDIM, TEST_NDIM)
#         self.X2 = self.S2 @ self.S2.T + np.eye(TEST_NDIM)
#         self.L1 = np.linalg.cholesky(self.X1)
#         self.L2 = np.linalg.cholesky(self.X2)
#         self.H = np.random.rand(TEST_NDIM, TEST_NDIM)
#
#     def test_yields_cholesky_both(self):
#         received = pnfs.cholesky_update(self.H @ self.L1, self.L2)
#         expected = np.linalg.cholesky(self.H @ self.X1 @ self.H.T + self.X2)
#         self.assertAllClose(received, expected)
#
#     def test_yields_cholesky_sum(self):
#         received = pnfs.cholesky_update(self.L1, self.L2)
#         expected = np.linalg.cholesky(self.X1 + self.X2)
#         self.assertAllClose(received, expected)
#
#     def test_yields_cholesky_prod(self):
#         received = pnfs.cholesky_update(self.H @ self.L1)
#         expected = np.linalg.cholesky(self.H @ self.X1 @ self.H.T)
#         self.assertAllClose(received, expected)
#
#
# if __name__ == "__main__":
#     unittest.main()
