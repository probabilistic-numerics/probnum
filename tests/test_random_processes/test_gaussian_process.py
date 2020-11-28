"""Test cases for Gaussian processes."""

import numpy as np

import probnum.utils as _utils
from probnum import random_processes as rps
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
                x = np.random.normal(size=(5,) + _utils.as_shape(gp.input_dim))
                gp_eval = gp(x)
                self.assertIsInstance(gp_eval, rvs.Normal)


class InstantiationTestCase(GaussianProcessTestCase):
    """Test instantiation of Gaussian processes."""

    def test_gp_from_covfunction_needs_dimensions(self):
        """Not providing in-/output dimensions for a custom kernel raises an error."""
        mean = lambda x: 0.0
        kern = lambda x0, x1=None: x0 @ x0.T if x1 is None else x0 @ x1.T
        with self.assertRaises(ValueError):
            rps.GaussianProcess(mean=mean, cov=kern)
