"""Test cases for random processes."""

import unittest

import numpy as np

from probnum import random_processes as rps
from tests.testing import NumpyAssertions


class RandomProcessTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for random variables."""

    def setUp(self) -> None:
        """Create different random processes for tests."""

        # Seed
        np.random.seed(42)

        # Mean functions
        def mean_zero(x, out_dim=None):
            if out_dim is not None:
                shape = (np.asarray(x).shape[0], out_dim)
            else:
                shape = np.asarray(x).shape[0]
            return np.zeros(shape=shape)

        self.mean_functions = [mean_zero]

        # Covariance functions
        cov_lin = lambda x0, x1: (x0 - 1.0) @ (x1 - 1.0).T
        cov_poly = lambda x0, x1: (x0 @ x1.T) ** 3
        cov_expquad = lambda x0, x1: np.exp(-0.5 * (x0 - x1) @ (x0 - x1).T)
        self.cov_functions = [cov_lin, cov_poly, cov_expquad]

        # Deterministic processes
        self.deterministic_processes = []

        # Gaussian processes
        self.gaussian_processes = [
            rps.GaussianProcess(
                mean=lambda x: mean_zero(x),
                cov=cov_lin,
                input_shape=(1,),
                output_shape=(),
            ),
            rps.GaussianProcess(
                mean=lambda x: mean_zero(x),
                cov=cov_poly,
                input_shape=(2,),
                output_shape=(),
            ),
            rps.GaussianProcess(
                mean=lambda x: mean_zero(x),
                cov=cov_expquad,
                input_shape=(3,),
                output_shape=(),
            ),
            rps.GaussianProcess(
                mean=lambda x: mean_zero(x, 2),
                cov=cov_poly,
                input_shape=(),
                output_shape=(2,),
            ),
        ]

        # Gauss-Markov processes
        self.gaussmarkov_processes = []

        # Generic random processes
        self.random_processes = (
            self.deterministic_processes
            + self.gaussian_processes
            + self.gaussmarkov_processes
        )


class InstantiationTestCase(RandomProcessTestCase):
    """Test random process instantiation"""

    def test_rp_from_function(self):
        """Create a random process from a function."""
        for fun in self.mean_functions:
            with self.subTest():
                randproc = rps.asrandproc(fun)
                self.assertIsInstance(randproc, rps.RandomProcess)


class ArithmeticTestCase(RandomProcessTestCase):
    """Test random process arithmetic"""


class ShapeTestCase(RandomProcessTestCase):
    """Test shapes of random process in-/output and associated functions."""

    def test_rp_output_shape(self):
        """
        Test whether output shape matches shape of evaluations of the random process.
        """
        for rand_proc in self.random_processes:
            with self.subTest():
                n_inputs_x = 10
                x = np.random.normal(size=(n_inputs_x,) + rand_proc.input_shape)
                y = rand_proc(x)
                self.assertTupleEqual(
                    tuple1=(x.shape[0],) + rand_proc.output_shape, tuple2=y.shape
                )

    def test_rp_mean_shape(self):
        """
        Test whether output shape matches the shape of the mean function of the random
        process.
        """
        for rand_proc in self.random_processes:
            with self.subTest():
                n_inputs_x = 10
                x = np.random.normal(size=(n_inputs_x,) + rand_proc.input_shape)
                meanfun = rand_proc.mean(x)
                self.assertTupleEqual(
                    tuple1=(x.shape[0],) + rand_proc.output_shape, tuple2=meanfun.shape
                )

    def test_rp_cov_shape(self):
        """
        Test whether the covariance at a set of n inputs has dimension n x n x
        d x d, where d is the output dimension of the process. In the common case of a
        process with one-dimensional output the covariance should have shape n x n.
        """
        for rand_proc in self.random_processes:
            with self.subTest():
                n_inputs_x = 10
                x = np.random.normal(size=(n_inputs_x,) + rand_proc.input_shape)
                covar = rand_proc.cov(x, x)
                self.assertTupleEqual(
                    tuple1=(x.shape[0], x.shape[0])
                    + rand_proc.output_shape
                    + rand_proc.output_shape,
                    tuple2=covar.shape,
                )

    def test_rp_var_shape(self):
        """
        Test whether output shape matches the shape of the variance function of the
        random process.
        """
        for rand_proc in self.random_processes:
            with self.subTest():
                n_inputs_x = 10
                x = np.random.normal(size=(n_inputs_x,) + rand_proc.input_shape)
                varfun = rand_proc.var(x)
                self.assertTupleEqual(
                    tuple1=(x.shape[0],) + rand_proc.output_shape, tuple2=varfun.shape
                )

    def test_rp_sample_shape(self):
        pass
