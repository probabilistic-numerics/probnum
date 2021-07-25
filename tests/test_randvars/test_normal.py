"""Tests for the normal distribution."""
import itertools
import unittest

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.stats

from probnum import config, linops, randvars
from probnum.problems.zoo.linalg import random_spd_matrix
from tests.testing import NumpyAssertions


class NormalTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for the normal distribution."""

    def setUp(self):
        """Resources for tests."""

        # Seed
        self.seed = 42
        self.rng = np.random.default_rng(seed=self.seed)

        # Parameters
        m = 7
        n = 3
        self.constants = [-1, -2.4, 0, 200, np.pi]
        sparsemat = scipy.sparse.rand(m=m, n=n, density=0.1, random_state=self.rng)
        self.normal_params = [
            # Univariate
            (-1.0, 3.0),
            (1, 3),
            # Multivariate
            (np.random.uniform(size=10), np.eye(10)),
            (np.random.uniform(size=10), random_spd_matrix(rng=self.rng, dim=10)),
            # Matrixvariate
            (
                np.random.uniform(size=(2, 2)),
                linops.SymmetricKronecker(
                    A=np.array([[1.0, 2.0], [2.0, 10.0]]),
                    B=np.array([[5.0, -1.0], [-1.0, 10.0]]),
                ).todense(),
            ),
            # Operatorvariate
            (
                np.array([1.0, -5.0]),
                linops.Matrix(A=np.array([[2.0, 1.0], [1.0, 1.0]])),
            ),
            (
                linops.Matrix(A=np.array([[0.0, -5.0]])),
                linops.Identity(shape=(2, 2)),
            ),
            (
                np.array([[1.0, 2.0], [-3.0, -0.4], [4.0, 1.0]]),
                linops.Kronecker(A=np.eye(3), B=5 * np.eye(2)),
            ),
            (
                linops.Matrix(A=sparsemat.todense()),
                linops.Kronecker(0.1 * linops.Identity(m), linops.Identity(n)),
            ),
            (
                linops.Matrix(A=np.random.uniform(size=(2, 2))),
                linops.SymmetricKronecker(
                    A=np.array([[1.0, 2.0], [2.0, 10.0]]),
                    B=np.array([[5.0, -1.0], [-1.0, 10.0]]),
                ),
            ),
            # Symmetric Kronecker Identical Factors
            (
                linops.Identity(shape=25),
                linops.SymmetricKronecker(A=linops.Identity(25)),
            ),
        ]

    def test_correct_instantiation(self):
        """Test whether different variants of the normal distribution are instances of
        Normal."""
        for mean, cov in self.normal_params:
            with self.subTest():
                dist = randvars.Normal(mean=mean, cov=cov)
                self.assertIsInstance(dist, randvars.Normal)

    def test_scalarmult(self):
        """Multiply a rv with a normal distribution with a scalar."""
        for (mean, cov), const in list(
            itertools.product(self.normal_params, self.constants)
        ):
            with self.subTest():
                normrv = const * randvars.Normal(mean=mean, cov=cov)

                self.assertIsInstance(normrv, randvars.RandomVariable)

                if const != 0:
                    self.assertIsInstance(normrv, randvars.Normal)
                else:
                    self.assertIsInstance(normrv, randvars.Constant)

    def test_addition_normal(self):
        """Add two random variables with a normal distribution."""
        for (mean0, cov0), (mean1, cov1) in list(
            itertools.product(self.normal_params, self.normal_params)
        ):
            with self.subTest():
                normrv0 = randvars.Normal(mean=mean0, cov=cov0)
                normrv1 = randvars.Normal(mean=mean1, cov=cov1)

                if normrv0.shape == normrv1.shape:
                    try:
                        newmean = mean0 + mean1
                        newcov = cov0 + cov1
                    except TypeError:
                        continue

                    self.assertIsInstance(normrv0 + normrv1, randvars.Normal)
                else:
                    with self.assertRaises(ValueError):
                        normrv_added = normrv0 + normrv1

    def test_rv_linop_kroneckercov(self):
        """Create a rv with a normal distribution with linear operator mean and
        Kronecker product kernels."""

        @linops.LinearOperator.broadcast_matvec
        def _matmul(v):
            return np.array([2 * v[0], 3 * v[1]])

        A = linops.LinearOperator(shape=(2, 2), dtype=np.double, matmul=_matmul)
        V = linops.Kronecker(A, A)
        randvars.Normal(mean=A, cov=V)

    def test_normal_dimension_mismatch(self):
        """Instantiating a normal distribution with mismatched mean and kernels should
        result in a ValueError."""
        for mean, cov in [
            (0, np.array([1, 2])),
            (np.array([1, 2]), np.array([1, 0])),
            (np.array([[-1, 0], [2, 1]]), np.eye(3)),
        ]:
            with self.subTest():
                err_msg = "Mean and kernels mismatch in normal distribution did not raise a ValueError."
                with self.assertRaises(ValueError, msg=err_msg):
                    assert randvars.Normal(mean=mean, cov=cov)

    def test_normal_instantiation(self):
        """Instantiation of a normal distribution with mixed mean and cov type."""
        for mean, cov in self.normal_params:
            with self.subTest():
                randvars.Normal(mean=mean, cov=cov)

    def test_normal_pdf(self):
        """Evaluate pdf at random input."""
        for mean, cov in self.normal_params:
            with self.subTest():
                dist = randvars.Normal(mean=mean, cov=cov)
                pass

    def test_normal_cdf(self):
        """Evaluate cdf at random input."""
        pass

    def test_sample(self):
        """Draw samples and check all sample dimensions."""
        for mean, cov in self.normal_params:
            with self.subTest():
                # TODO: check dimension of each realization in rv_sample
                rv = randvars.Normal(mean=mean, cov=cov)
                rv_sample = rv.sample(rng=self.rng, size=5)
                if not np.isscalar(rv.mean):
                    self.assertEqual(
                        rv_sample.shape[-rv.ndim :],
                        mean.shape,
                        msg="Realization shape does not match mean shape.",
                    )

    def test_sample_zero_cov(self):
        """Draw sample from distribution with zero kernels and check whether it equals
        the mean."""
        for mean, cov in self.normal_params:
            with self.subTest():
                rv = randvars.Normal(mean=mean, cov=0 * cov)
                rv_sample = rv.sample(rng=self.rng, size=1)
                assert_str = "Draw with kernels zero does not match mean."
                if isinstance(rv.mean, linops.LinearOperator):
                    self.assertAllClose(rv_sample, rv.mean.todense(), msg=assert_str)
                else:
                    self.assertAllClose(rv_sample, rv.mean, msg=assert_str)

    def test_symmetric_samples(self):
        """Samples from a normal distribution with symmetric Kronecker kernels of two
        symmetric matrices are symmetric."""

        n = 3
        A = self.rng.uniform(size=(n, n))
        A = 0.5 * (A + A.T) + n * np.eye(n)
        rv = randvars.Normal(
            mean=np.eye(A.shape[0]),
            cov=linops.SymmetricKronecker(A=A),
        )
        rv = rv.sample(rng=self.rng, size=10)
        for i, B in enumerate(rv):
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
        """Indexing with Python integers yields a univariate normal distribution."""
        for mean, cov in self.normal_params:
            rv = randvars.Normal(mean=mean, cov=cov)

            with self.subTest():
                # Sample random index
                idx = tuple(np.random.randint(dim_size) for dim_size in rv.shape)

                # Index into distribution
                indexed_rv = rv[idx]

                self.assertIsInstance(indexed_rv, randvars.Normal)

                # Compare with expected parameter values
                if rv.ndim == 0:
                    flat_idx = ()
                elif rv.ndim == 1:
                    flat_idx = (idx[0],)
                else:
                    assert rv.ndim == 2

                    flat_idx = (idx[0] * rv.shape[1] + idx[1],)

                self.assertEqual(indexed_rv.shape, ())
                self.assertEqual(indexed_rv.mean, rv.dense_mean[idx])
                self.assertEqual(indexed_rv.var, rv.var[idx])
                self.assertEqual(indexed_rv.cov, rv.dense_cov[flat_idx + flat_idx])

    def test_slicing(self):
        """Slicing into a normal distribution yields a normal distribution of the same
        type."""
        for mean, cov in self.normal_params:
            rv = randvars.Normal(mean=mean, cov=cov)

            def _random_slice(dim_size):
                start = np.random.randint(0, dim_size)
                stop = np.random.randint(start + 1, dim_size + 1)

                return slice(start, stop)

            with self.subTest():
                # Sample random slice objects for each dimension
                slices = tuple(_random_slice(dim_size) for dim_size in rv.shape)

                # Get slice from distribution
                sliced_rv = rv[slices]

                # Compare with expected parameter values
                slice_mask = np.zeros_like(rv.dense_mean, dtype=np.bool_)
                slice_mask[slices] = True
                slice_mask = slice_mask.ravel()

                self.assertArrayEqual(sliced_rv.mean, rv.dense_mean[slices])
                self.assertArrayEqual(sliced_rv.var, rv.var[slices])

                if rv.ndim > 0:
                    self.assertArrayEqual(
                        sliced_rv.cov, rv.dense_cov[np.ix_(slice_mask, slice_mask)]
                    )
                else:
                    self.assertArrayEqual(sliced_rv.cov, rv.cov)

    def test_array_indexing(self):
        """Indexing with 1-dim integer arrays yields a multivariate normal."""
        for mean, cov in self.normal_params:
            rv = randvars.Normal(mean=mean, cov=cov)

            if rv.ndim == 0:
                continue

            with self.subTest():
                # Sample random indices
                idcs = tuple(
                    np.random.randint(dim_shape, size=10) for dim_shape in mean.shape
                )

                # Index into distribution
                indexed_rv = rv[idcs]

                self.assertIsInstance(indexed_rv, randvars.Normal)

                # Compare with expected parameter values
                if len(rv.shape) == 1:
                    flat_idcs = idcs[0]
                else:
                    assert len(rv.shape) == 2

                    flat_idcs = idcs[0] * rv.shape[1] + idcs[1]

                self.assertEqual(indexed_rv.shape, (10,))

                self.assertArrayEqual(indexed_rv.mean, rv.dense_mean[idcs])
                self.assertArrayEqual(indexed_rv.var, rv.var[idcs])
                self.assertArrayEqual(
                    indexed_rv.cov, rv.dense_cov[np.ix_(flat_idcs, flat_idcs)]
                )

    def test_array_indexing_broadcast(self):
        """Indexing with broadcasted integer arrays yields a matrixvariate normal."""
        for mean, cov in self.normal_params:
            rv = randvars.Normal(mean=mean, cov=cov)

            if rv.ndim != 2:
                continue

            with self.subTest():
                # Sample random indices
                idcs = np.ix_(
                    *tuple(
                        np.random.randint(dim_shape, size=10) for dim_shape in rv.shape
                    )
                )

                # Index into distribution
                indexed_rv = rv[idcs]

                self.assertIsInstance(indexed_rv, randvars.Normal)
                self.assertEqual(indexed_rv.shape, (10, 10))

                # Compare with expected parameter values
                flat_idcs = np.broadcast_arrays(*idcs)
                flat_idcs = flat_idcs[0] * rv.shape[1] + flat_idcs[1]
                flat_idcs = flat_idcs.ravel()

                self.assertArrayEqual(indexed_rv.mean, rv.dense_mean[idcs])
                self.assertArrayEqual(indexed_rv.var, rv.var[idcs])
                self.assertArrayEqual(
                    indexed_rv.cov, rv.dense_cov[np.ix_(flat_idcs, flat_idcs)]
                )

    def test_masking(self):
        """Masking a multivariate or matrixvariate normal yields a multivariate
        normal."""
        for mean, cov in self.normal_params:
            rv = randvars.Normal(mean=mean, cov=cov)

            with self.subTest():
                # Sample random indices
                idcs = tuple(
                    np.random.randint(dim_shape, size=10) for dim_shape in rv.shape
                )

                mask = np.zeros_like(rv.dense_mean, dtype=np.bool_)
                mask[idcs] = True

                # Mask distribution
                masked_rv = rv[mask]

                self.assertIsInstance(masked_rv, randvars.Normal)

                # Compare with expected parameter values
                flat_mask = mask.flatten()

                self.assertArrayEqual(masked_rv.mean, rv.dense_mean[mask])
                self.assertArrayEqual(masked_rv.var, rv.var[mask])

                if rv.ndim == 0:
                    self.assertArrayEqual(masked_rv.cov, rv.cov)
                else:
                    self.assertArrayEqual(
                        masked_rv.cov, rv.dense_cov[np.ix_(flat_mask, flat_mask)]
                    )


class UnivariateNormalTestCase(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        self.seed = 42
        self.rng = np.random.default_rng(seed=self.seed)
        self.params = (self.rng.uniform(), self.rng.gamma(shape=6, scale=1.0))

    def test_reshape_newaxis(self):
        dist = randvars.Normal(*self.params)

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

                    self.assertEqual(newdist.shape, newshape)
                    self.assertEqual(np.squeeze(newdist.mean), dist.mean)
                    self.assertEqual(np.squeeze(newdist.cov), dist.cov)

    def test_transpose(self):
        dist = randvars.Normal(*self.params)
        dist_t = dist.transpose()

        self.assertArrayEqual(dist_t.mean, dist.mean)
        self.assertArrayEqual(dist_t.cov, dist.cov)

        # Test sampling
        fixed_rng = np.random.default_rng(seed=self.seed)
        dist_sample = dist.sample(rng=fixed_rng, size=5)

        fixed_rng = np.random.default_rng(seed=self.seed)
        dist_t_sample = dist_t.sample(rng=fixed_rng, size=5)

        self.assertArrayEqual(dist_t_sample, dist_sample)

    def test_cov_cholesky_cov_cholesky_not_passed(self):
        """No cov_cholesky is passed in init.

        In this case, the "is_precomputed" flag is False, a cov_cholesky
        is computed on demand, but can also be computed manually with
        any damping factor.
        """
        mean, cov = self.params
        rv = randvars.Normal(mean, cov)

        with self.subTest("No Cholesky precomputed"):
            self.assertFalse(rv.cov_cholesky_is_precomputed)

        with self.subTest("Cholesky factor is computed correctly"):
            # The default damping factor 1e-12 does not mess up this test
            self.assertAllClose(rv.cov_cholesky, np.sqrt(rv.cov))

        with self.subTest("Cholesky is precomputed"):
            self.assertTrue(rv.cov_cholesky_is_precomputed)

    def test_precompute_cov_cholesky(self):
        mean, cov = self.params
        rv = randvars.Normal(mean, cov)

        with self.subTest("No Cholesky precomputed"):
            self.assertFalse(rv.cov_cholesky_is_precomputed)

        with self.subTest("Damping factor check"):

            rv.precompute_cov_cholesky(damping_factor=10.0)
            self.assertAllClose(rv.cov_cholesky, np.sqrt(rv.cov + 10.0))

        with self.subTest("Cholesky is precomputed"):
            self.assertTrue(rv.cov_cholesky_is_precomputed)

    def test_damping_factor_config(self):
        mean, cov = self.params
        rv = randvars.Normal(mean, cov)

        chol_standard_damping = rv.dense_cov_cholesky(damping_factor=None)
        self.assertAllClose(
            chol_standard_damping,
            np.sqrt(rv.cov + 1e-12),
        )

        with config(covariance_inversion_damping=1e-3):
            chol_altered_damping = rv.dense_cov_cholesky(damping_factor=None)
            self.assertAllClose(
                chol_altered_damping,
                np.sqrt(rv.cov + 1e-3),
            )

    def test_cov_cholesky_cov_cholesky_passed(self):
        """A value for cov_cholesky is passed in init.

        In this case, the "is_precomputed" flag is True, the
        cov_cholesky returns the argument that has been passed, but
        (p)recomputing overwrites the argument with a new factor.
        """
        mean, cov = self.params

        # This is purposely not the correct Cholesky factor for test reasons
        cov_cholesky = np.random.rand()

        rv = randvars.Normal(mean, cov, cov_cholesky=cov_cholesky)

        with self.subTest("Cholesky precomputed"):
            self.assertTrue(rv.cov_cholesky_is_precomputed)

        with self.subTest("Returns correct cov_cholesky"):
            self.assertAllClose(rv.cov_cholesky, cov_cholesky)

        with self.subTest("self.precompute raises exception"):
            with self.assertRaises(Exception):
                rv.precompute_cov_cholesky()


class MultivariateNormalTestCase(unittest.TestCase, NumpyAssertions):
    def setUp(self):

        self.seed = 42
        self.rng = np.random.default_rng(self.seed)

        self.params = (
            self.rng.uniform(size=10),
            random_spd_matrix(rng=self.rng, dim=10),
        )

    def test_newaxis(self):
        vector_rv = randvars.Normal(*self.params)

        matrix_rv = vector_rv[:, np.newaxis]

        self.assertEqual(matrix_rv.shape, (10, 1))
        self.assertArrayEqual(np.squeeze(matrix_rv.mean), vector_rv.mean)
        self.assertArrayEqual(matrix_rv.cov, vector_rv.cov)

    def test_reshape(self):
        rv = randvars.Normal(*self.params)

        newshape = (5, 2)
        reshaped_rv = rv.reshape(newshape)

        self.assertArrayEqual(reshaped_rv.mean, rv.mean.reshape(newshape))
        self.assertArrayEqual(reshaped_rv.cov, rv.cov)

        # Test sampling
        fixed_rng = np.random.default_rng(seed=self.seed)
        dist_sample = rv.sample(rng=fixed_rng, size=5)

        fixed_rng = np.random.default_rng(seed=self.seed)
        dist_reshape_sample = reshaped_rv.sample(rng=fixed_rng, size=5)

        self.assertArrayEqual(
            dist_reshape_sample, dist_sample.reshape((-1,) + newshape)
        )

    def test_transpose(self):
        rv = randvars.Normal(*self.params)
        transposed_rv = rv.transpose()

        self.assertArrayEqual(transposed_rv.mean, rv.mean)
        self.assertArrayEqual(transposed_rv.cov, rv.cov)

        # Test sampling
        fixed_rng = np.random.default_rng(seed=self.seed)
        dist_sample = rv.sample(rng=fixed_rng, size=5)
        fixed_rng = np.random.default_rng(seed=self.seed)
        dist_t_sample = transposed_rv.sample(rng=fixed_rng, size=5)

        self.assertArrayEqual(dist_t_sample, dist_sample)

    def test_cov_cholesky_cov_cholesky_not_passed(self):
        """No cov_cholesky is passed in init.

        In this case, the "is_precomputed" flag is False, a cov_cholesky
        is computed on demand, but can also be computed manually with
        any damping factor.
        """
        mean, cov = self.params

        rv = randvars.Normal(mean, cov)

        with self.subTest("No Cholesky precomputed"):
            self.assertFalse(rv.cov_cholesky_is_precomputed)

        with self.subTest("Cholesky factor is computed correctly"):
            # The default damping factor 1e-12 does not mess up this test
            self.assertAllClose(rv.cov_cholesky, np.linalg.cholesky(rv.cov))

        with self.subTest("Cholesky is precomputed"):
            self.assertTrue(rv.cov_cholesky_is_precomputed)

    def test_precompute_cov_cholesky(self):
        mean, cov = self.params
        rv = randvars.Normal(mean, cov)

        with self.subTest("No Cholesky precomputed"):
            self.assertFalse(rv.cov_cholesky_is_precomputed)

        with self.subTest("Damping factor check"):
            with config(lazy_linalg=False):
                rv.precompute_cov_cholesky(damping_factor=10.0)
                self.assertIsInstance(rv.cov_cholesky, np.ndarray)
                self.assertAllClose(
                    rv.cov_cholesky,
                    np.linalg.cholesky(rv.cov + 10.0 * np.eye(len(rv.cov))),
                )

        with self.subTest("Cholesky is precomputed"):
            self.assertTrue(rv.cov_cholesky_is_precomputed)

    def test_precompute_cov_cholesky_with_linops(self):
        mean, cov = self.params
        rv = randvars.Normal(mean, linops.aslinop(cov))

        with self.subTest("No Cholesky precomputed"):
            self.assertFalse(rv.cov_cholesky_is_precomputed)

        with self.subTest("Damping factor check"):
            with config(lazy_linalg=True):
                rv.precompute_cov_cholesky(damping_factor=10.0)
                self.assertIsInstance(rv.cov_cholesky, linops.LinearOperator)
                self.assertAllClose(
                    rv.cov_cholesky.todense(),
                    np.linalg.cholesky(cov + 10.0 * np.eye(rv.cov.shape[0])),
                )

        with self.subTest("Cholesky is precomputed"):
            self.assertTrue(rv.cov_cholesky_is_precomputed)

    def test_cov_cholesky_cov_cholesky_passed(self):
        """A value for cov_cholesky is passed in init.

        In this case, the "is_precomputed" flag is True, the
        cov_cholesky returns the argument that has been passed, but
        (p)recomputing overwrites the argument with a new factor.
        """
        mean, cov = self.params

        # This is purposely not the correct Cholesky factor for test reasons
        cov_cholesky = np.random.rand(*cov.shape)

        rv = randvars.Normal(mean, cov, cov_cholesky=cov_cholesky)

        with self.subTest("Cholesky precomputed"):
            self.assertTrue(rv.cov_cholesky_is_precomputed)

        with self.subTest("Returns correct cov_cholesky"):
            self.assertAllClose(rv.cov_cholesky, cov_cholesky)

        with self.subTest("self.precompute raises exception"):
            with self.assertRaises(Exception):
                rv.precompute_cov_cholesky()

    def test_cholesky_cov_incompatible_types(self):
        """Test the behaviour of Normal.__init__ in the setup where the type of the
        Cholesky factor and the type of the covariance do not match."""
        mean, cov = self.params
        cov_cholesky = np.linalg.cholesky(cov)
        cov_cholesky_wrong_type = cov_cholesky.tolist()
        with self.subTest("Different type raises ValueError"):
            with self.assertRaises(TypeError):
                randvars.Normal(mean, cov, cov_cholesky=cov_cholesky_wrong_type)

        cov_cholesky_wrong_shape = cov_cholesky[1:]
        with self.subTest("Different shape raises ValueError"):
            with self.assertRaises(ValueError):
                randvars.Normal(mean, cov, cov_cholesky=cov_cholesky_wrong_shape)

        cov_cholesky_wrong_dtype = cov_cholesky.astype(int)
        with self.subTest("Different data type is promoted"):

            # Sanity check
            self.assertNotEqual(cov.dtype, cov_cholesky_wrong_dtype.dtype)

            # Assert data type of cov_cholesky is changed during __init__
            normal_new_dtype = randvars.Normal(
                mean, cov, cov_cholesky=cov_cholesky_wrong_dtype
            )
            self.assertEqual(
                normal_new_dtype.cov.dtype, normal_new_dtype.cov_cholesky.dtype
            )


class MatrixvariateNormalTestCase(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        # Seed
        self.seed = 42
        self.rng = np.random.default_rng(seed=self.seed)

    def test_reshape(self):
        rv = randvars.Normal(
            mean=np.random.uniform(size=(4, 3)),
            cov=linops.Kronecker(
                A=random_spd_matrix(rng=self.rng, dim=4),
                B=random_spd_matrix(rng=self.rng, dim=3),
            ).todense(),
        )

        newshape = (2, 6)
        reshaped_rv = rv.reshape(newshape)

        self.assertArrayEqual(reshaped_rv.mean, rv.mean.reshape(newshape))
        self.assertArrayEqual(reshaped_rv.cov, rv.cov)

        # Test sampling
        fixed_rng = np.random.default_rng(seed=self.seed)
        dist_sample = rv.sample(rng=fixed_rng, size=5)
        fixed_rng = np.random.default_rng(seed=self.seed)
        dist_reshape_sample = reshaped_rv.sample(rng=fixed_rng, size=5)

        self.assertArrayEqual(
            dist_reshape_sample, dist_sample.reshape((-1,) + newshape)
        )

    def test_transpose(self):
        rv = randvars.Normal(
            mean=np.random.uniform(size=(2, 2)),
            cov=random_spd_matrix(rng=self.rng, dim=4),
        )
        transposed_rv = rv.transpose()

        self.assertArrayEqual(transposed_rv.mean, rv.mean.T)

        # Test covariance
        for ii, ij in itertools.product(range(2), range(2)):
            for ji, jj in itertools.product(range(2), range(2)):
                idx = (2 * ii + ij, 2 * ji + jj)
                idx_t = (2 * ij + ii, 2 * jj + ji)

                self.assertEqual(transposed_rv.cov[idx_t], rv.cov[idx])

        # Sadly, sampling is not stable w.r.t. permutations of variables

    def test_cov_cholesky_cov_cholesky_not_passed(self):
        """No cov_cholesky is passed in init.

        In this case, the "is_precomputed" flag is False, a cov_cholesky
        is computed on demand, but can also be computed manually with
        any damping factor.
        """
        rv = randvars.Normal(
            mean=np.random.uniform(size=(2, 2)),
            cov=random_spd_matrix(rng=self.rng, dim=4),
        )

        with self.subTest("No Cholesky precomputed"):
            self.assertFalse(rv.cov_cholesky_is_precomputed)

        with self.subTest("Cholesky factor is computed correctly"):
            # The default damping factor 1e-12 does not mess up this test
            self.assertAllClose(rv.cov_cholesky, np.linalg.cholesky(rv.cov))

        with self.subTest("Cholesky is precomputed"):
            self.assertTrue(rv.cov_cholesky_is_precomputed)

    def test_precompute_cov_cholesky(self):
        rv = randvars.Normal(
            mean=np.random.uniform(size=(2, 2)),
            cov=random_spd_matrix(rng=self.rng, dim=4),
        )

        with self.subTest("No Cholesky precomputed"):
            self.assertFalse(rv.cov_cholesky_is_precomputed)

        with self.subTest("Damping factor check"):
            rv.precompute_cov_cholesky(damping_factor=10.0)
            self.assertAllClose(
                rv.cov_cholesky, np.linalg.cholesky(rv.cov + 10.0 * np.eye(len(rv.cov)))
            )

        with self.subTest("Cholesky is precomputed"):
            self.assertTrue(rv.cov_cholesky_is_precomputed)

    def test_cov_cholesky_cov_cholesky_passed(self):
        """A value for cov_cholesky is passed in init.

        In this case, the "is_precomputed" flag is True, the
        cov_cholesky returns the argument that has been passed, but
        (p)recomputing overwrites the argument with a new factor.
        """
        # This is purposely not the correct Cholesky factor for test reasons
        cov_cholesky = np.random.rand(4, 4)

        rv = randvars.Normal(
            mean=np.random.uniform(size=(2, 2)),
            cov=random_spd_matrix(rng=self.rng, dim=4),
            cov_cholesky=cov_cholesky,
        )

        with self.subTest("Cholesky precomputed"):
            self.assertTrue(rv.cov_cholesky_is_precomputed)

        with self.subTest("Returns correct cov_cholesky"):
            self.assertAllClose(rv.cov_cholesky, cov_cholesky)

        with self.subTest("self.precompute raises exception"):
            with self.assertRaises(Exception):
                rv.precompute_cov_cholesky()


if __name__ == "__main__":
    unittest.main()
