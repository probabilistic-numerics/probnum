"""Tests for the normal distribution."""
import itertools
import unittest

import numpy as np
import scipy.sparse
import scipy.stats

from probnum import prob
from probnum.linalg import linops
from tests.testing import NumpyAssertions


def _random_spd_matrix(D=10):
    """ Generates a random symmetric positive definite matrix. """

    # Sample a rotation matrix Q in SO(D) (the special orthogonal group SO(D), or
    # orthogonal matrices with unit determinant, drawn uniformly from the Haar measure.
    #
    # The algorithm used is the subgroup algorithm as originally proposed by
    #
    # P. Diaconis & M. Shahshahani, "The subgroup algorithm for generating uniform
    # random variables". Probability in the Engineering and Informational Sciences 1:
    # 15?32 (1987)
    t = 2 * np.pi * np.random.rand()
    A = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])

    for d in range(2, D):
        v = np.random.randn(d + 1, 1)
        v /= np.sqrt(v.T.dot(v))
        e = np.zeros((d + 1, 1))
        e[0] += 1

        x = e - v
        x /= np.sqrt(x.T.dot(x))

        D_ = np.zeros((d + 1, d + 1))
        D_[0, 0] += 1
        D_[1:, 1:] += A
        A = D_ - 2 * x * (x.T.dot(D_))

    Q = -A

    # Generate random positive spectrum
    lam = np.random.gamma(shape=6, scale=1.0, size=D)

    return Q @ np.diag(lam) @ Q.T


class NormalTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for the normal distribution."""

    def setUp(self):
        """Resources for tests."""
        # Seed
        np.random.seed(seed=42)

        # Parameters
        m = 7
        n = 3
        self.constants = [-1, -2.4, 0, 200, np.pi]
        sparsemat = scipy.sparse.rand(m=m, n=n, density=0.1, random_state=1)
        self.normal_params = [
            # Univariate
            (-1, 3),
            # Multivariate
            (np.random.uniform(size=10), np.eye(10)),
            (np.random.uniform(size=10), _random_spd_matrix(10)),
            # Matrixvariate
            (
                np.random.uniform(size=(2, 2)),
                linops.SymmetricKronecker(
                    A=np.array([[1, 2], [2, 1]]), B=np.array([[5, -1], [-1, 10]])
                ).todense(),
            ),
            # Operatorvariate
            (np.array([1, -5]), linops.MatrixMult(A=np.array([[2, 1], [1, -0.1]]))),
            (linops.MatrixMult(A=np.array([[0, -5]])), linops.Identity(shape=(2, 2))),
            (
                np.array([[1, 2], [-3, -0.4], [4, 1]]),
                linops.Kronecker(A=np.eye(3), B=5 * np.eye(2)),
            ),
            (
                linops.MatrixMult(A=sparsemat.todense()),
                linops.Kronecker(0.1 * linops.Identity(m), linops.Identity(n)),
            ),
            (
                linops.MatrixMult(A=np.random.uniform(size=(2, 2))),
                linops.SymmetricKronecker(
                    A=np.array([[1, 2], [2, 1]]), B=np.array([[5, -1], [-1, 10]])
                ),
            ),
            # Symmetric Kronecker Identical Factors
            (
                linops.Identity(shape=25),
                linops.SymmetricKronecker(A=linops.Identity(25)),
            ),
        ]

    def test_correct_instantiation(self):
        """Test whether different variants of the normal distribution are instances of Normal."""
        for mean, cov in self.normal_params:
            with self.subTest():
                dist = prob.Normal(mean=mean, cov=cov)
                self.assertIsInstance(dist, prob.Normal)

    def test_scalarmult(self):
        """Multiply a rv with a normal distribution with a scalar."""
        for (mean, cov), const in list(
            itertools.product(self.normal_params, self.constants)
        ):
            with self.subTest():
                normrv = const * prob.RandomVariable(
                    distribution=prob.Normal(mean=mean, cov=cov)
                )
                self.assertIsInstance(normrv, prob.RandomVariable)
                if const != 0:
                    self.assertIsInstance(normrv.distribution, prob.Normal)
                else:
                    self.assertIsInstance(normrv.distribution, prob.Dirac)

    def test_addition_normal(self):
        """Add two random variables with a normal distribution"""
        for (mean0, cov0), (mean1, cov1) in list(
            itertools.product(self.normal_params, self.normal_params)
        ):
            with self.subTest():
                normrv0 = prob.RandomVariable(
                    distribution=prob.Normal(mean=mean0, cov=cov0)
                )
                normrv1 = prob.RandomVariable(
                    distribution=prob.Normal(mean=mean1, cov=cov1)
                )

                if not isinstance(normrv1.distribution, type(normrv0.distribution)):
                    continue

                if normrv0.shape == normrv1.shape:
                    self.assertIsInstance((normrv0 + normrv1).distribution, prob.Normal)
                else:
                    with self.assertRaises(TypeError):
                        normrv_added = normrv0 + normrv1

    def test_rv_linop_kroneckercov(self):
        """Create a rv with a normal distribution with linear operator mean and Kronecker product kernels."""

        def mv(v):
            return np.array([2 * v[0], 3 * v[1]])

        A = linops.LinearOperator(shape=(2, 2), matvec=mv)
        V = linops.Kronecker(A, A)
        prob.RandomVariable(distribution=prob.Normal(mean=A, cov=V))

    def test_normal_dimension_mismatch(self):
        """Instantiating a normal distribution with mismatched mean and kernels should result in a ValueError."""
        for mean, cov in [
            (0, [1, 2]),
            (np.array([1, 2]), np.array([1, 0])),
            (np.array([[-1, 0], [2, 1]]), np.eye(3)),
        ]:
            with self.subTest():
                err_msg = "Mean and kernels mismatch in normal distribution did not raise a ValueError."
                with self.assertRaises(ValueError, msg=err_msg):
                    assert prob.Normal(mean=mean, cov=cov)

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
                    self.assertEqual(
                        dist_sample.shape[-ndims_rv:],
                        mean.shape,
                        msg="Realization shape does not match mean shape.",
                    )

    def test_sample_zero_cov(self):
        """Draw sample from distribution with zero kernels and check whether it equals the mean."""
        for mean, cov in self.normal_params:
            with self.subTest():
                dist = prob.Normal(mean=mean, cov=0 * cov, random_state=1)
                dist_sample = dist.sample(size=1)
                assert_str = "Draw with kernels zero does not match mean."
                if isinstance(dist.mean(), linops.LinearOperator):
                    self.assertAllClose(
                        dist_sample, dist.mean().todense(), msg=assert_str
                    )
                else:
                    self.assertAllClose(dist_sample, dist.mean(), msg=assert_str)

    def test_symmetric_samples(self):
        """Samples from a normal distribution with symmetric Kronecker kernels of two symmetric matrices are
        symmetric."""
        np.random.seed(42)
        n = 3
        A = np.random.uniform(size=(n, n))
        A = 0.5 * (A + A.T) + n * np.eye(n)
        dist = prob.Normal(
            mean=np.eye(A.shape[0]), cov=linops.SymmetricKronecker(A=A), random_state=1
        )
        dist_sample = dist.sample(size=10)
        for i, B in enumerate(dist_sample):
            self.assertAllClose(
                B,
                B.T,
                atol=1e-5,
                rtol=1e-5,
                msg="Sample {} from symmetric Kronecker distribution is not symmetric.".format(
                    i
                ),
            )

    def test_indexing(self):
        """ Indexing with Python integers yields a univariate normal distribution. """
        for mean, cov in self.normal_params:
            dist = prob.Normal(mean=mean, cov=cov)

            if isinstance(
                dist,
                (
                    prob.distributions.normal._UnivariateNormal,
                    prob.distributions.normal._OperatorvariateNormal,  # TODO: Implement slicing on linear operators
                ),
            ):
                continue

            with self.subTest():
                # Sample random index
                idx = tuple(np.random.randint(dim_size) for dim_size in dist.shape)

                # Index into distribution
                index_dist = dist[idx]

                self.assertIsInstance(
                    index_dist, prob.distributions.normal._UnivariateNormal
                )

                # Compare with expected parameter values
                if len(dist.shape) == 1:
                    flat_idx = idx[0]
                else:
                    assert len(dist.shape) == 2

                    flat_idx = idx[0] * dist.shape[1] + idx[1]

                self.assertEqual(index_dist.mean(), dist.mean()[idx])
                self.assertEqual(index_dist.var(), dist.var()[flat_idx])
                self.assertEqual(index_dist.cov(), dist.cov()[flat_idx, flat_idx])

    def test_slicing(self):
        """ Slicing into a normal distribution yields a normal distribution of the same type """
        for mean, cov in self.normal_params:
            dist = prob.Normal(mean=mean, cov=cov)

            if isinstance(
                dist,
                (
                    prob.distributions.normal._UnivariateNormal,
                    prob.distributions.normal._OperatorvariateNormal,  # TODO: Implement slicing on linear operators
                ),
            ):
                continue

            def _random_slice(dim_size):
                start = np.random.randint(0, dim_size)
                stop = np.random.randint(start + 1, dim_size + 1)

                return slice(start, stop)

            with self.subTest():
                # Sample random slice objects for each dimension
                slices = tuple(_random_slice(dim_size) for dim_size in dist.shape)

                # Get slice from distribution
                sliced_dist = dist[slices]

                self.assertIsInstance(sliced_dist, type(dist))

                # Compare with expected parameter values
                slice_mask = np.zeros_like(dist.mean(), dtype=np.bool)
                slice_mask[slices] = True
                slice_mask = slice_mask.ravel()

                self.assertArrayEqual(sliced_dist.mean(), dist.mean()[slices])
                self.assertArrayEqual(sliced_dist.var(), dist.var()[slice_mask])
                self.assertArrayEqual(
                    sliced_dist.cov(), dist.cov()[np.ix_(slice_mask, slice_mask)]
                )

    def test_array_indexing(self):
        """ Indexing with 1-dim integer arrays yields a multivariate normal. """
        for mean, cov in self.normal_params:
            dist = prob.Normal(mean=mean, cov=cov)

            if isinstance(
                dist,
                (
                    prob.distributions.normal._UnivariateNormal,
                    prob.distributions.normal._OperatorvariateNormal,
                ),
            ):
                continue

            with self.subTest():
                # Sample random indices
                idcs = tuple(
                    np.random.randint(dim_shape, size=10) for dim_shape in mean.shape
                )

                # Index into distribution
                index_dist = dist[idcs]

                self.assertIsInstance(
                    index_dist, prob.distributions.normal._MultivariateNormal
                )

                # Compare with expected parameter values
                if len(dist.shape) == 1:
                    flat_idcs = idcs[0]
                else:
                    assert len(dist.shape) == 2

                    flat_idcs = idcs[0] * dist.shape[1] + idcs[1]

                self.assertEqual(index_dist.shape, (10,))

                self.assertArrayEqual(index_dist.mean(), dist.mean()[idcs])
                self.assertArrayEqual(index_dist.var(), dist.var()[flat_idcs])
                self.assertArrayEqual(
                    index_dist.cov(), dist.cov()[np.ix_(flat_idcs, flat_idcs)]
                )

    def test_array_indexing_broadcast(self):
        """ Indexing with broadcasted integer arrays yields a matrixvariate normal. """
        for mean, cov in self.normal_params:
            dist = prob.Normal(mean=mean, cov=cov)

            if isinstance(
                dist,
                (
                    prob.distributions.normal._UnivariateNormal,
                    prob.distributions.normal._MultivariateNormal,
                    prob.distributions.normal._OperatorvariateNormal,
                ),
            ):
                continue

            assert len(dist.shape) == 2

            with self.subTest():
                # Sample random indices
                idcs = np.ix_(
                    *tuple(
                        np.random.randint(dim_shape, size=10)
                        for dim_shape in dist.shape
                    )
                )

                # Index into distribution
                index_dist = dist[idcs]

                self.assertIsInstance(
                    index_dist, prob.distributions.normal._MatrixvariateNormal
                )
                self.assertEqual(index_dist.shape, (10, 10))

                # Compare with expected parameter values
                flat_idcs = np.broadcast_arrays(*idcs)
                flat_idcs = flat_idcs[0] * dist.shape[1] + flat_idcs[1]
                flat_idcs = flat_idcs.ravel()

                self.assertArrayEqual(index_dist.mean(), dist.mean()[idcs])
                self.assertArrayEqual(index_dist.var(), dist.var()[flat_idcs])
                self.assertArrayEqual(
                    index_dist.cov(), dist.cov()[np.ix_(flat_idcs, flat_idcs)]
                )

    def test_masking(self):
        """ Masking a multivariate or matrixvariate normal yields a multivariate normal. """
        for mean, cov in self.normal_params:
            dist = prob.Normal(mean=mean, cov=cov)

            if isinstance(
                dist,
                (
                    prob.distributions.normal._UnivariateNormal,
                    prob.distributions.normal._OperatorvariateNormal,
                ),
            ):
                continue

            with self.subTest():
                # Sample random indices
                idcs = tuple(
                    np.random.randint(dim_shape, size=10) for dim_shape in mean.shape
                )

                mask = np.zeros_like(dist.mean(), dtype=np.bool)
                mask[idcs] = True

                # Mask distribution
                index_dist = dist[mask]

                self.assertIsInstance(
                    index_dist, prob.distributions.normal._MultivariateNormal
                )

                # Compare with expected parameter values
                flat_mask = mask.flatten()

                self.assertArrayEqual(index_dist.mean(), dist.mean()[mask])
                self.assertArrayEqual(index_dist.var(), dist.var()[flat_mask])
                self.assertArrayEqual(
                    index_dist.cov(), dist.cov()[np.ix_(flat_mask, flat_mask)]
                )


class UnivariateNormalTestCase(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        self.params = (np.random.uniform(), np.random.gamma(shape=6, scale=1.0))

    def test_reshape_newaxis(self):
        dist = prob.Normal(*self.params)

        for ndim in range(1, 3):
            for method in ["newaxis", "reshape"]:
                with self.subTest():
                    newshape = tuple([1] * ndim)

                    if method == "newaxis":
                        if ndim == 1:
                            newdist = dist[np.newaxis]
                        elif ndim == 2:
                            newdist = dist[np.newaxis, np.newaxis]
                    elif method == "reshape":
                        newdist = dist.reshape(newshape)

                    self.assertIsInstance(
                        newdist,
                        prob.distributions.normal._MultivariateNormal
                        if ndim == 1
                        else prob.distributions.normal._MatrixvariateNormal,
                    )

                    self.assertEqual(newdist.shape, newshape)
                    self.assertEqual(np.squeeze(newdist.mean()), dist.mean())
                    self.assertEqual(np.squeeze(newdist.cov()), dist.cov())

    def test_transpose(self):
        dist = prob.Normal(*self.params)
        dist_t = dist.transpose()

        self.assertArrayEqual(dist_t.mean(), dist.mean())
        self.assertArrayEqual(dist_t.cov(), dist.cov())

        # Test sampling
        dist.random_state = 42
        dist_sample = dist.sample(size=5)

        dist_t.random_state = 42
        dist_t_sample = dist_t.sample(size=5)

        self.assertArrayEqual(dist_t_sample, dist_sample)


class MultivariateNormalTestCase(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        self.params = (np.random.uniform(size=10), _random_spd_matrix(10))

    def test_newaxis(self):
        dist = prob.Normal(*self.params)

        matrix_dist = dist[:, np.newaxis]

        self.assertIsInstance(
            matrix_dist, prob.distributions.normal._MatrixvariateNormal
        )

        self.assertEqual(matrix_dist.shape, (10, 1))
        self.assertArrayEqual(np.squeeze(matrix_dist.mean()), dist.mean())
        self.assertArrayEqual(matrix_dist.cov(), dist.cov())

    def test_reshape(self):
        dist = prob.Normal(*self.params)

        newshape = (5, 2)
        dist_reshape = dist.reshape(newshape)

        self.assertArrayEqual(dist_reshape.mean(), dist.mean().reshape(newshape))
        self.assertArrayEqual(dist_reshape.cov(), dist.cov())

        # Test sampling
        dist.random_state = 42
        dist_sample = dist.sample(size=5)

        dist_reshape.random_state = 42
        dist_reshape_sample = dist_reshape.sample(size=5)

        self.assertArrayEqual(
            dist_reshape_sample, dist_sample.reshape((-1,) + newshape)
        )

    def test_transpose(self):
        dist = prob.Normal(*self.params)
        dist_t = dist.transpose()

        self.assertArrayEqual(dist_t.mean(), dist.mean())
        self.assertArrayEqual(dist_t.cov(), dist.cov())

        # Test sampling
        dist.random_state = 42
        dist_sample = dist.sample(size=5)

        dist_t.random_state = 42
        dist_t_sample = dist_t.sample(size=5)

        self.assertArrayEqual(dist_t_sample, dist_sample)


class MatrixvariateNormalTestCase(unittest.TestCase, NumpyAssertions):
    def test_reshape(self):
        dist = prob.Normal(
            mean=np.random.uniform(size=(4, 3)),
            cov=linops.Kronecker(
                A=_random_spd_matrix(4), B=_random_spd_matrix(3)
            ).todense(),
        )

        newshape = (2, 6)
        dist_reshape = dist.reshape(newshape)

        self.assertArrayEqual(dist_reshape.mean(), dist.mean().reshape(newshape))
        self.assertArrayEqual(dist_reshape.cov(), dist.cov())

        # Test sampling
        dist.random_state = 42
        dist_sample = dist.sample(size=5)

        dist_reshape.random_state = 42
        dist_reshape_sample = dist_reshape.sample(size=5)

        self.assertArrayEqual(
            dist_reshape_sample, dist_sample.reshape((-1,) + newshape)
        )

    def test_transpose(self):
        dist = prob.Normal(
            mean=np.random.uniform(size=(2, 2)), cov=_random_spd_matrix(4),
        )
        dist_t = dist.transpose()

        self.assertArrayEqual(dist_t.mean(), dist.mean().T)

        # Test covariance
        for ii, ij in itertools.product(range(2), range(2)):
            for ji, jj in itertools.product(range(2), range(2)):
                idx = (2 * ii + ij, 2 * ji + jj)
                idx_t = (2 * ij + ii, 2 * jj + ji)

                self.assertEqual(dist_t.cov()[idx_t], dist.cov()[idx])

        # Sadly, sampling is not stable w.r.t. permutations of variables


if __name__ == "__main__":
    unittest.main()
