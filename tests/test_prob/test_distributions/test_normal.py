"""Tests for the normal distribution."""

import unittest
from tests.testing import NumpyAssertions

import numpy as np
import scipy.sparse

from probnum import prob
from probnum.linalg import linops


class NormalTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for the normal distribution."""

    def setUp(self):
        """Resources for tests."""
        # Seed
        np.random.seed(seed=42)

        # Parameters
        m = 7
        n = 3
        sparsemat = scipy.sparse.rand(m=m, n=n, density=0.1, random_state=1)
        self.normal_params = [
            (-1, 3),
            (np.random.uniform(size=10), np.eye(10)),
            (np.array([1, -5]), linops.MatrixMult(A=np.array([[2, 1], [1, -.1]]))),
            (linops.MatrixMult(A=np.array([[0, -5]])), linops.Identity(shape=(2, 2))),
            (np.array([[1, 2], [-3, -.4], [4, 1]]), linops.Kronecker(A=np.eye(3), B=5 * np.eye(2))),
            (linops.MatrixMult(A=sparsemat.todense()),
             linops.Kronecker(0.1 * linops.Identity(m), linops.Identity(n))),
            (linops.MatrixMult(A=np.random.uniform(size=(2, 2))),
             linops.SymmetricKronecker(A=np.array([[1, 2], [2, 1]]), B=np.array([[5, -1], [-1, 10]]))),
            (linops.Identity(shape=25), linops.SymmetricKronecker(A=linops.Identity(25)))
        ]

    def test_rv_linop_kroneckercov(self):
        """Create a rv with a normal distribution with linear operator mean and Kronecker product covariance."""

        def mv(v):
            return np.array([2 * v[0], 3 * v[1]])

        A = linops.LinearOperator(shape=(2, 2), matvec=mv)
        V = linops.Kronecker(A, A)
        prob.RandomVariable(distribution=prob.Normal(mean=A, cov=V))

    def test_normal_dimension_mismatch(self):
        """Instantiating a normal distribution with mismatched mean and covariance should result in a ValueError."""
        for mean, cov in [(0, [1, 2]),
                          (np.array([1, 2]), np.array([1, 0])),
                          (np.array([[-1, 0], [2, 1]]), np.eye(3))]:
            with self.subTest():
                with self.assertRaises(ValueError):
                    assert prob.Normal(mean=mean,
                                       cov=cov), "Mean and covariance mismatch in normal distribution."

    def test_normal_instantiation(self):
        """Instantiation of a normal distribution with mixed mean and cov type."""
        for mean, cov in self.normal_params:
            with self.subTest():
                prob.Normal(mean=mean, cov=cov)

    def test_normal_pdf(self):
        """Evaluate pdf at random input."""
        for mean, cov in self.normal_params:
            with self.subTest():
                dist = prob.Normal(mean=mean, cov=cov)
                pass

    def test_normal_cdf(self):
        """Evaluate cdf at random input."""
        pass

    def test_sample(self):
        """Draw samples and check all sample dimensions."""
        for mean, cov in self.normal_params:
            with self.subTest():
                # TODO: check dimension of each realization in dist_sample
                dist = prob.Normal(mean=mean, cov=cov, random_state=1)
                dist_sample = dist.sample(size=5)
                if not np.isscalar(dist.mean()):
                    ndims_rv = len(mean.shape)
                    self.assertEqual(dist_sample.shape[-ndims_rv:], mean.shape,
                                     msg="Realization shape does not match mean shape.")

    def test_sample_zero_cov(self):
        """Draw sample from distribution with zero covariance and check whether it equals the mean."""
        for mean, cov in self.normal_params:
            with self.subTest():
                dist = prob.Normal(mean=mean, cov=0 * cov, random_state=1)
                dist_sample = dist.sample(size=1)
                assert_str = "Draw with covariance zero does not match mean."
                if isinstance(dist.mean(), linops.LinearOperator):
                    self.assertAllClose(dist_sample, dist.mean().todense(), msg=assert_str)
                else:
                    self.assertAllClose(dist_sample, dist.mean(), msg=assert_str)

    def test_symmetric_samples(self):
        """Samples from a normal distribution with symmetric Kronecker covariance of two symmetric matrices are
        symmetric."""
        np.random.seed(42)
        n = 3
        A = np.random.uniform(size=(n, n))
        A = 0.5 * (A + A.T) + n * np.eye(n)
        dist = prob.Normal(mean=np.eye(A.shape[0]),
                           cov=linops.SymmetricKronecker(A=A), random_state=1)
        dist_sample = dist.sample(size=10)
        for i, B in enumerate(dist_sample):
            self.assertAllClose(B, B.T, atol=1e-5, rtol=1e-5,
                                msg="Sample {} from symmetric Kronecker distribution is not symmetric.".format(i))


if __name__ == "__main__":
    unittest.main()
