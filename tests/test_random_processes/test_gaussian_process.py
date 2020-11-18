"""Test cases for Gaussian processes."""

import numpy as np

from probnum import random_variables as rvs

from .test_random_process import RandomProcessTestCase


class GaussianProcessTestCase(RandomProcessTestCase):
    """General test case for Gaussian processes."""


class PropertiesTestCase(GaussianProcessTestCase):
    """Test known properties of Gaussian processes."""

    def test_finite_evaluation_is_normal(self):
        """A Gaussian process evaluated at a finite set of inputs is a Gaussian random
        variable."""
        for gp in self.gaussian_processes:
            with self.subTest():
                x = np.random.normal(size=(5,) + gp.input_shape)
                gp_eval = gp(x)
                self.assertIsInstance(gp_eval, rvs.Normal)
