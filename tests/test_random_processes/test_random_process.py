"""Test cases for random processes."""

import unittest

import numpy as np

import probnum.utils as _utils
from probnum import random_processes as rps
from tests.testing import NumpyAssertions


class RandomProcessTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for random processes."""

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
        def cov_lin(x0, x1, keepdims=False, constant=1.0):
            """Linear kernel."""
            covmat = np.sum(
                np.multiply.outer(x0 - constant, x1 - constant), axis=(1, 3)
            )
            if keepdims:
                return covmat
            else:
                return covmat.reshape(x0.shape[0], x1.shape[0])

        def cov_poly(x0, x1, power=3, keepdims=False):
            """Polynomial kernel."""
            covmat = np.sum(np.multiply.outer(x0, x1), axis=(1, 3)) ** power
            if keepdims:
                return covmat
            else:
                return covmat.reshape(x0.shape[0], x1.shape[0])

        def cov_expquad(x0, x1, keepdims=False):
            """Exponentiated quadratic kernel."""
            covmat = np.exp(-0.5 * np.sum(np.subtract.outer(x0, x1) ** 2, axis=(1, 3)))
            if keepdims:
                return covmat
            else:
                return covmat.reshape(x0.shape[0], x1.shape[0])

        def cov_coreg_expquad(x0, x1, keepdims=False):
            """Coregionalization kernel multiplied with an RBF kernel."""
            covmat = np.multiply.outer(
                cov_expquad(x0, x1, keepdims=False), np.array([[4, 2], [2, 1]])
            )
            if keepdims:
                return covmat
            else:
                return np.transpose(covmat, axes=[2, 0, 3, 1]).reshape(
                    2 * x0.shape[0], 2 * x1.shape[0]
                )

        self.cov_functions = [cov_lin, cov_poly, cov_expquad]

        # Deterministic processes
        self.deterministic_processes = [
            rps.DeterministicProcess(
                fun=mean_zero, input_dim=1, output_dim=1, dtype=np.dtype(np.float_)
            )
        ]

        # Gaussian processes
        self.gaussian_processes = [
            rps.GaussianProcess(
                mean=lambda x: mean_zero(x),
                cov=cov_lin,
                input_dim=(1,),
                output_dim=(),
            ),
            rps.GaussianProcess(
                mean=lambda x: mean_zero(x),
                cov=cov_poly,
                input_dim=(2, 1),
                output_dim=(),
            ),
            rps.GaussianProcess(
                mean=lambda x: mean_zero(x),
                cov=cov_expquad,
                input_dim=3,
                output_dim=1,
            ),
            rps.GaussianProcess(
                mean=lambda x: mean_zero(x, out_dim=2),
                cov=cov_coreg_expquad,
                input_dim=(),
                output_dim=(2,),
            ),
        ]

        # Gauss-Markov processes
        self.gaussmarkov_processes = []
        self.gaussian_processes += self.gaussmarkov_processes

        # Generic random processes
        self.random_processes = (
            self.deterministic_processes
            + self.gaussian_processes
            + self.gaussmarkov_processes
        )


class InstantiationTestCase(RandomProcessTestCase):
    """Test random process instantiation."""

    def test_rp_from_function(self):
        """Create a random process from a function."""
        for fun in self.mean_functions:
            with self.subTest():
                randproc = rps.asrandproc(fun)
                self.assertIsInstance(randproc, rps.RandomProcess)


class ArithmeticTestCase(RandomProcessTestCase):
    """Test random process arithmetic."""


class ShapeTestCase(RandomProcessTestCase):
    """Test shapes of random process in-/output and associated functions."""

    def test_rp_output_shape(self):
        """Test whether output shape matches shape of evaluations of the random
        process."""
        for rand_proc in self.random_processes:
            with self.subTest():
                n_inputs_x = 10
                x = np.random.normal(size=(n_inputs_x,) + rand_proc.input_shape)
                y = rand_proc(x)
                if rand_proc.output_shape == () or rand_proc.output_shape == (1,):
                    rand_proc_out_shape = ()
                else:
                    rand_proc_out_shape = rand_proc.output_shape

                self.assertTupleEqual(
                    tuple1=(x.shape[0],) + rand_proc_out_shape,
                    tuple2=y.shape,
                    msg=f"Output of {repr(rand_proc)} does not have the "
                    f"correct shape.",
                )

    def test_rp_mean_shape(self):
        """Test whether output shape matches the shape of the mean function of the
        random process."""
        for rand_proc in self.random_processes:
            with self.subTest():
                n_inputs_x = 10
                x = np.random.normal(size=(n_inputs_x,) + rand_proc.input_shape)
                meanfun = rand_proc.mean(x)
                if rand_proc.output_shape == () or rand_proc.output_shape == (1,):
                    rand_proc_out_shape = ()
                else:
                    rand_proc_out_shape = rand_proc.output_shape

                self.assertTupleEqual(
                    tuple1=(x.shape[0],) + rand_proc_out_shape,
                    tuple2=meanfun.shape,
                    msg=f"Mean of {repr(rand_proc)} does not have the "
                    f"correct shape.",
                )

    def test_rp_cov_shape(self):
        """Test whether the covariance at a set of n inputs has dimension n x n x d x d,
        where d is the output dimension of the process.

        In the common case of a process with one-dimensional output the
        covariance should have shape n x n.
        """
        for rand_proc in self.random_processes:
            with self.subTest():
                n_inputs_x = 10
                x = np.random.normal(size=(n_inputs_x,) + rand_proc.input_shape)
                covar_matrix = rand_proc.cov(x, x, flatten=False)
                covar_tensor = rand_proc.cov(x, x, flatten=True)

                self.assertTupleEqual(
                    tuple1=(
                        x.shape[0] * np.prod(rand_proc.output_shape),
                        x.shape[0] * np.prod(rand_proc.output_shape),
                    ),
                    tuple2=covar_matrix.shape,
                    msg=f"Covariance of {repr(rand_proc)} shaped as a matrix does not "
                    f"have the correct shape.",
                )

                self.assertTupleEqual(
                    tuple1=(x.shape[0], x.shape[0])
                    + rand_proc.output_shape
                    + rand_proc.output_shape,
                    tuple2=covar_tensor.shape,
                    msg=f"Covariance of {repr(rand_proc)} shaped as a tensor does not "
                    "have the correct shape.",
                )

    def test_rp_var_shape(self):
        """Test whether output shape matches the shape of the variance function of the
        random process."""
        for rand_proc in self.random_processes:
            with self.subTest():
                n_inputs_x = 10
                x = np.random.normal(size=(n_inputs_x,) + rand_proc.input_shape)
                varfun = rand_proc.var(x)

                if rand_proc.output_shape == () or rand_proc.output_shape == (1,):
                    rand_proc_out_shape = ()
                else:
                    rand_proc_out_shape = rand_proc.output_shape

                self.assertTupleEqual(
                    tuple1=(x.shape[0],) + rand_proc_out_shape, tuple2=varfun.shape
                )

    def test_rp_sample_shape(self):
        """Test whether output shape matches last dimensions of a drawn sample from the
        process."""
        for rand_proc in self.random_processes:
            for sample_size in [(), 1, (2,), (2, 2)]:
                with self.subTest():
                    n_inputs_x = 10
                    x = np.random.normal(size=(n_inputs_x,) + rand_proc.input_shape)
                    sample_paths = rand_proc.sample(x, size=sample_size)
                    self.assertTupleEqual(
                        tuple1=sample_paths.shape,
                        tuple2=(n_inputs_x,)
                        + _utils.as_shape(sample_size)
                        + rand_proc.output_shape,
                        msg=f"Samples from {repr(rand_proc)} do not have the correct "
                        f"shape.",
                    )


class MethodTestCase(RandomProcessTestCase):
    """Test the methods of a random process."""

    def test_samples_are_callables(self):
        """When not specifying inputs to the sample method it should return ``size``
        number of callables."""
        for rand_proc in self.random_processes:
            with self.subTest():
                self.assertTrue(callable(rand_proc.sample(size=())))
