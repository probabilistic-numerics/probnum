"""Test cases for the rational quadratic kernel."""

import probnum.kernels as kerns

from .test_kernel import KernelTestCase


class RationalQuadraticTestCase(KernelTestCase):
    """Test case for rational quadratic kernels."""

    def test_nonpositive_alpha_raises_exception(self):
        """Check whether a non-positive alpha parameter raises a ValueError."""
        for alpha in [-1, -1.0, 0.0, 0]:
            with self.subTest():
                with self.assertRaises(ValueError):
                    kerns.RatQuad(input_dim=1, alpha=alpha)
