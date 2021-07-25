"""Tests for constant random variables."""
import unittest

import numpy as np

from probnum import config, linops, randvars


def test_constant_accessible_like_gaussian():
    """Constant has an attribute cov_cholesky which returns zeros."""
    support = np.array([2, 3])
    s = randvars.Constant(support)
    np.testing.assert_allclose(s.cov, s.cov_cholesky)


class TestConstant(unittest.TestCase):
    """General test case for constant random variables."""

    def setUp(self):
        self.supports = [1, np.array([1, 2]), np.array([[0]]), np.array([[6], [-0.3]])]

        # A random number generator needs to be passed to any sample()
        # function, but for Constant rvs, this is ignored.
        self.unimportant_rng = np.random.default_rng(seed=1)

    def test_logpdf(self):
        pass

    def test_cov_linops(self):
        # Ignore scalar support, because in this case, LinOps make no sense.
        for supp in self.supports[1:]:
            with self.subTest():
                with config(lazy_linalg=True):
                    rv_lo = randvars.Constant(support=supp)
                    assert isinstance(rv_lo.cov, linops.LinearOperator)
                with config(lazy_linalg=False):
                    rv_de = randvars.Constant(support=supp)
                    assert isinstance(rv_de.cov, np.ndarray)

    def test_sample_shapes(self):
        """Test whether samples have the correct shapes."""
        for supp in self.supports:
            for sample_size in [1, (), 10, (4,), (3, 2)]:
                with self.subTest():
                    s = randvars.Constant(support=supp).sample(
                        rng=self.unimportant_rng, size=sample_size
                    )
                    if sample_size == ():
                        self.assertEqual(np.shape(supp), np.shape(s))
                    elif isinstance(sample_size, tuple):
                        self.assertEqual(sample_size + np.shape(supp), np.shape(s))
                    else:
                        self.assertEqual(
                            tuple([sample_size, *np.shape(supp)]), np.shape(s)
                        )


if __name__ == "__main__":
    unittest.main()
