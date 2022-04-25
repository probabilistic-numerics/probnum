"""Tests for the normal distribution."""
import itertools
import unittest

import numpy as np
import scipy.sparse
import scipy.stats

from probnum import config, linops, randvars
from probnum.problems.zoo.linalg import random_spd_matrix

from tests.testing import NumpyAssertions


class NormalTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for the normal distribution."""

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

        In this case, the "is_precomputed" flag is False, a cov_cholesky is computed on
        demand, but can also be computed manually with any damping factor.
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

    def test_cov_cholesky_cov_cholesky_passed(self):
        """A value for cov_cholesky is passed in init.

        In this case, the "is_precomputed" flag is True, the cov_cholesky returns the
        argument that has been passed, but (p)recomputing overwrites the argument with a
        new factor.
        """
        mean, cov = self.params

        # This is purposely not the correct Cholesky factor for test reasons
        cov_cholesky = np.random.rand()

        rv = randvars.Normal(mean, cov, cache={"cov_cholesky": cov_cholesky})

        with self.subTest("Cholesky precomputed"):
            self.assertTrue(rv.cov_cholesky_is_precomputed)

        with self.subTest("Returns correct cov_cholesky"):
            self.assertAllClose(rv.cov_cholesky, cov_cholesky)

        with self.subTest("self.precompute raises exception"):
            with self.assertRaises(Exception):
                rv.precompute_cov_cholesky()


class MultivariateNormalTestCase(unittest.TestCase, NumpyAssertions):
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

        In this case, the "is_precomputed" flag is False, a cov_cholesky is computed on
        demand, but can also be computed manually with any damping factor.
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
            with config(matrix_free=False):
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
            with config(matrix_free=True):
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

        In this case, the "is_precomputed" flag is True, the cov_cholesky returns the
        argument that has been passed, but (p)recomputing overwrites the argument with a
        new factor.
        """
        mean, cov = self.params

        # This is purposely not the correct Cholesky factor for test reasons
        cov_cholesky = np.random.rand(*cov.shape)

        rv = randvars.Normal(mean, cov, cache={"cov_cholesky": cov_cholesky})

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
                randvars.Normal(
                    mean, cov, cache={"cov_cholesky": cov_cholesky_wrong_type}
                )

        cov_cholesky_wrong_shape = cov_cholesky[1:]
        with self.subTest("Different shape raises ValueError"):
            with self.assertRaises(ValueError):
                randvars.Normal(
                    mean, cov, cache={"cov_cholesky": cov_cholesky_wrong_shape}
                )

        cov_cholesky_wrong_dtype = cov_cholesky.astype(int)
        with self.subTest("Different data type is promoted"):

            # Sanity check
            self.assertNotEqual(cov.dtype, cov_cholesky_wrong_dtype.dtype)

            # Assert data type of cov_cholesky is changed during __init__
            normal_new_dtype = randvars.Normal(
                mean, cov, cache={"cov_cholesky": cov_cholesky_wrong_dtype}
            )
            self.assertEqual(
                normal_new_dtype.cov.dtype, normal_new_dtype.cov_cholesky.dtype
            )


class MatrixvariateNormalTestCase(unittest.TestCase, NumpyAssertions):
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

        In this case, the "is_precomputed" flag is False, a cov_cholesky is computed on
        demand, but can also be computed manually with any damping factor.
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

        In this case, the "is_precomputed" flag is True, the cov_cholesky returns the
        argument that has been passed, but (p)recomputing overwrites the argument with a
        new factor.
        """
        # This is purposely not the correct Cholesky factor for test reasons
        cov_cholesky = np.random.rand(4, 4)

        rv = randvars.Normal(
            mean=np.random.uniform(size=(2, 2)),
            cov=random_spd_matrix(rng=self.rng, dim=4),
            cache={"cov_cholesky": cov_cholesky},
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
