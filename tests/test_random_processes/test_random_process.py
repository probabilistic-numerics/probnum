"""Test cases for random processes."""

import unittest

import numpy as np

import probnum.utils as _utils
from probnum import kernels as kerns
from probnum import random_processes as rps
from probnum import random_variables as rvs
from tests.testing import NumpyAssertions


class RandomProcessTestCase(unittest.TestCase, NumpyAssertions):
    """General test case for random processes."""

    def setUp(self) -> None:
        """Create different random processes for tests."""

        # Random number generator
        self.rng = np.random.default_rng(seed=42)

        # Mean functions
        def mean_zero(x, out_dim=1):
            x = np.asarray(x)
            if x.ndim > 1:
                shape = (x.shape[0], out_dim)
            else:
                shape = (out_dim,)

            return np.zeros(shape=shape)

        self.mean_functions = [mean_zero]

        # Covariance functions
        self.cov_functions = [
            (kerns.Kernel, {"kernelfun": lambda x0, x1: np.inner(x0, x1).squeeze()}),
            (kerns.Linear, {"shift": 1.0}),
            (kerns.WhiteNoise, {"sigma": -1.0}),
            (kerns.Polynomial, {"constant": 1.0, "exponent": 3}),
            (kerns.ExpQuad, {"lengthscale": 1.5}),
            (kerns.RatQuad, {"lengthscale": 0.5, "alpha": 2.0}),
        ]

        # Deterministic processes
        self.deterministic_processes = [
            rps.DeterministicProcess(
                fun=lambda x: np.zeros((x.shape[0], 1)) if x.ndim > 1 else np.zeros(1)
            ),
        ]

        # Gaussian processes
        self.gaussian_processes = [
            rps.GaussianProcess(
                mean=mean_zero, cov=kerns.Linear(shift=1.0, input_dim=1)
            ),
            rps.GaussianProcess(
                mean=mean_zero,
                cov=kerns.Polynomial(exponent=3, constant=0.5, input_dim=2),
            ),
            rps.GaussianProcess(
                mean=mean_zero, cov=kerns.ExpQuad(lengthscale=1.5, input_dim=3)
            ),
            # rps.GaussianProcess(
            #     mean=lambda x: mean_zero(x, out_dim=2),
            #     cov=cov_coreg_expquad,
            #     input_dim=1,
            #     output_dim=2,
            # ),
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

        # Fix random state of all random processes
        for rp in self.random_processes:
            rp.random_state = 42


class InstantiationTestCase(RandomProcessTestCase):
    """Test random process instantiation."""

    def test_rp_from_function(self):
        """Create a random process from a function."""
        for fun in self.mean_functions:
            with self.subTest():
                randproc = rps.asrandproc(fun)
                self.assertIsInstance(randproc, rps.RandomProcess)

    def test_rp_from_covariance_callable(self):
        """Create a random process with a covariance function from a callable."""
        kern = lambda x0, x1=None: x0 @ x0.T if x1 is None else x0 @ x1.T
        fun = rps.RandomProcess(input_dim=1, output_dim=1, cov=kern, dtype=np.float_)
        fun.cov(np.linspace(0, 1, 5)[:, None])

    def test_dimension_process_kernel_mismatch(self):
        """Test whether an error is raised if kernel and process dimension do not
        match."""
        with self.assertRaises(ValueError):
            kern = kerns.Linear(input_dim=5)
            rps.RandomProcess(input_dim=10, output_dim=1, cov=kern, dtype=np.float_)

    def test_covariance_not_callable(self):
        """Test whether an error is raised if the covariance is not a callable."""
        with self.assertRaises(TypeError):
            kern = 1.0
            rps.RandomProcess(input_dim=1, output_dim=1, cov=kern, dtype=np.float_)


class ArithmeticTestCase(RandomProcessTestCase):
    """Test random process arithmetic."""


class ShapeTestCase(RandomProcessTestCase):
    """Test shapes of random process in-/output and associated functions."""

    def _generic_shape_assert(self, x0, rand_proc, fun):
        """A generic shape test for functions of a random process taking one input."""
        if rand_proc.input_dim == 1:
            self.assertEqual(
                0,
                fun(x0[0, 0]).ndim,
                msg=f"Output of {repr(rand_proc)} for scalar input should "
                f"have 0 dimensions.",
            )

        if rand_proc.output_dim == 1:
            self.assertEqual(
                0,
                fun(x0[0, :]).ndim,
                msg=f"Output of {repr(rand_proc)} for vector input should "
                f"have 0 dimensions.",
            )
        else:
            x1 = x0[0, :]
            y1 = fun(x1)
            self.assertTupleEqual(
                tuple1=(rand_proc.output_dim,),
                tuple2=y1.shape,
                msg=f"Output of {repr(rand_proc)} for vector input should be a "
                f"vector.",
            )

        self.assertTupleEqual(
            tuple1=(x0.shape[0], rand_proc.output_dim),
            tuple2=fun(x0).shape,
            msg=f"Output of {repr(rand_proc)} does not have the "
            f"correct shape for multiple inputs.",
        )

    def test_output_shape(self):
        """Test whether evaluations of the random process have shape=(output_shape,) for
        an input vector or shape=(n, output_shape) for multiple inputs."""
        for rand_proc in self.random_processes:
            with self.subTest():
                n_inputs_x0 = 10
                x0 = self.rng.normal(size=(n_inputs_x0, rand_proc.input_dim))
                self._generic_shape_assert(x0=x0, rand_proc=rand_proc, fun=rand_proc)

    def test_mean_shape(self):
        """Test whether output shape matches the shape of the mean function of the
        random process."""
        for rand_proc in self.random_processes:
            with self.subTest():
                n_inputs_x = 10
                x0 = self.rng.normal(size=(n_inputs_x, rand_proc.input_dim))
                self._generic_shape_assert(
                    x0=x0, rand_proc=rand_proc, fun=rand_proc.mean
                )

    def test_var_shape(self):
        """Test whether output shape matches the shape of the variance function of the
        random process."""
        for rand_proc in self.random_processes:
            with self.subTest():
                # Data
                n_inputs_x0 = 10
                x0 = self.rng.normal(size=(n_inputs_x0, rand_proc.input_dim))
                self._generic_shape_assert(
                    x0=x0, rand_proc=rand_proc, fun=rand_proc.var
                )

    def test_std_shape(self):
        """Test whether output shape matches the shape of the standard deviation
        function of the random process."""
        for rand_proc in self.random_processes:
            with self.subTest():
                # Data
                n_inputs_x0 = 10
                x0 = self.rng.normal(size=(n_inputs_x0, rand_proc.input_dim))
                self._generic_shape_assert(
                    x0=x0, rand_proc=rand_proc, fun=rand_proc.std
                )

    def test_cov_shape(self):
        """Test whether the covariance function when evaluated has the correct shape."""
        for rand_proc in self.random_processes:
            with self.subTest():
                # Data
                n_inputs_x0 = 10
                n_inputs_x1 = 15
                x0 = self.rng.normal(size=(n_inputs_x0, rand_proc.input_dim))
                x1 = self.rng.normal(size=(n_inputs_x1, rand_proc.input_dim))

                # Input: (input_dim,), (input_dim,) -- Output: (output_dim, output_dim)
                out_shape = ()
                if rand_proc.output_dim > 1:
                    out_shape += (rand_proc.output_dim, rand_proc.output_dim)
                self.assertTupleEqual(
                    tuple1=rand_proc.cov(x0[0, :], x0[0, :]).shape,
                    tuple2=out_shape,
                    msg=f"Covariance of {repr(rand_proc)} does not have the correct "
                    f"shape for vector input.",
                )

                # Input: (n0, input_dim), (n1, input_dim) -- Output: (n0, n1)
                # or if output_dim > 1: (n0, n1, output_dim, output_dim)
                out_shape = (n_inputs_x0, n_inputs_x1)
                if rand_proc.output_dim > 1:
                    out_shape += (rand_proc.output_dim, rand_proc.output_dim)
                self.assertTupleEqual(
                    tuple1=rand_proc.cov(x0, x1).shape,
                    tuple2=out_shape,
                    msg=f"Covariance of {repr(rand_proc)} does not have the correct "
                    f"shape for multiple inputs.",
                )

    def test_sample_shape(self):
        """Test whether output shape matches last dimensions of a drawn sample from the
        process."""
        for rand_proc in self.random_processes:
            for sample_size in [(), 1, (2,), (2, 2)]:
                with self.subTest():
                    # Data
                    n_inputs_x = 10
                    x = self.rng.normal(size=(n_inputs_x, rand_proc.input_dim))

                    # Sample paths
                    sample_paths = rand_proc.sample(x, size=sample_size)
                    sample_size = _utils.as_shape(sample_size)
                    if len(sample_size) > 0 and sample_size[0] > 1:
                        sample_shape = sample_size
                    else:
                        sample_shape = ()
                    sample_shape += (
                        n_inputs_x,
                        rand_proc.output_dim,
                    )

                    self.assertTupleEqual(
                        tuple1=sample_paths.shape,
                        tuple2=sample_shape,
                        msg=f"Samples from {repr(rand_proc)} do not have the correct "
                        f"shape.",
                    )


class MethodTestCase(RandomProcessTestCase):
    """Test the methods of a random process."""

    def test_output_is_random_variable(self):
        """Test whether evaluating a random process returns a random variable."""
        for rand_proc in self.random_processes:
            with self.subTest():
                n_inputs_x0 = 10
                x0 = np.random.normal(size=(n_inputs_x0, rand_proc.input_dim))
                y0 = rand_proc(x0)

                self.assertIsInstance(
                    y0,
                    rvs.RandomVariable,
                    msg=f"Output of {repr(rand_proc)} is not a " f"random variable.",
                )

    def test_samples_are_callables(self):
        """When not specifying inputs to the sample method it should return ``size``
        number of callables."""
        for rand_proc in self.random_processes:
            with self.subTest():
                self.assertTrue(callable(rand_proc.sample(size=())))

    def test_rp_mean_cov_evaluated_matches_rv_mean_cov(self):
        """Check whether the evaluated mean and covariance function of a random process
        is equivalent to the mean and covariance of the evaluated random process as a
        random variable."""
        for rand_proc in self.random_processes:
            with self.subTest():
                x = np.random.normal(size=(10, rand_proc.input_dim))

                self.assertAllClose(
                    rand_proc(x).mean,
                    rand_proc.mean(x),
                    msg=f"Mean of evaluated {repr(rand_proc)} does not match the "
                    f"random process mean function evaluated.",
                )

                self.assertAllClose(
                    rand_proc(x).cov,
                    rand_proc.cov(x),
                    msg=f"Covariance of evaluated {repr(rand_proc)} does not match the "
                    f"random process mean function evaluated.",
                )
