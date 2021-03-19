"""Tests for random processes."""

from typing import Callable

import numpy as np
import pytest

from probnum import kernels, randprocs, randvars

# Random process instantiation


def test_rp_from_covariance_callable():
    """Create a random process with a covariance function from a callable."""
    kern = lambda x0, x1=None: x0 @ x0.T if x1 is None else x0 @ x1.T
    fun = randprocs.RandomProcess(input_dim=1, output_dim=1, cov=kern, dtype=np.float_)
    fun.cov(np.linspace(0, 1, 5)[:, None])


def test_dimension_process_kernel_mismatch():
    """Test whether an error is raised if kernel and process dimension do not match."""
    with pytest.raises(ValueError):
        kern = kernels.Linear(input_dim=5)
        randprocs.RandomProcess(input_dim=10, output_dim=1, cov=kern, dtype=np.float_)


def test_covariance_not_callable():
    """Test whether an error is raised if the covariance is not a callable."""
    with pytest.raises(TypeError):
        kern = 1.0
        randprocs.RandomProcess(input_dim=1, output_dim=1, cov=kern, dtype=np.float_)


# Random process shape


def generic_shape_assert(
    x0: np.ndarray, rand_proc: randprocs.RandomProcess, fun: Callable
):
    """A generic shape test for functions of a random process taking one input."""
    if rand_proc.input_dim == 1:
        assert (
            0 == fun(x0[0, 0]).ndim
        ), f"Output of {repr(rand_proc)} for scalar input should have 0 dimensions."

    if rand_proc.output_dim == 1:
        assert (
            0 == fun(x0[0, :]).ndim
        ), f"Output of {repr(rand_proc)} for vector input should have 0 dimensions."
    else:
        x1 = x0[0, :]
        y1 = fun(x1)
        assert (
            rand_proc.output_dim,
        ) == y1.shape, (
            f"Output of {repr(rand_proc)} for vector input should be a vector."
        )

    assert (x0.shape[0], rand_proc.output_dim) == fun(
        x0
    ).shape, f"Output of {repr(rand_proc)} does not have the correct shape for multiple inputs."


def test_output_shape(
    random_process: randprocs.RandomProcess, random_state: np.random.RandomState
):
    """Test whether evaluations of the random process have shape=(output_shape,) for an
    input vector or shape=(n, output_shape) for multiple inputs."""
    x0 = random_state.normal(size=(10, random_process.input_dim))
    generic_shape_assert(x0=x0, rand_proc=random_process, fun=random_process)


def test_mean_shape(
    random_process: randprocs.RandomProcess, random_state: np.random.RandomState
):
    """Test whether output shape matches the shape of the mean function of the random
    process."""
    x0 = random_state.normal(size=(10, random_process.input_dim))
    generic_shape_assert(x0=x0, rand_proc=random_process, fun=random_process.mean)


def test_var_shape(
    random_process: randprocs.RandomProcess, random_state: np.random.RandomState
):
    """Test whether output shape matches the shape of the variance function of the
    random process."""
    x0 = random_state.normal(size=(10, random_process.input_dim))
    generic_shape_assert(x0=x0, rand_proc=random_process, fun=random_process.var)


def test_std_shape(
    random_process: randprocs.RandomProcess, random_state: np.random.RandomState
):
    """Test whether output shape matches the shape of the standard deviation function of
    the random process."""
    x0 = random_state.normal(size=(10, random_process.input_dim))
    generic_shape_assert(x0=x0, rand_proc=random_process, fun=random_process.std)


# Random process methods


def test_evaluated_random_process_is_random_variable(
    random_process: randprocs.RandomProcess, random_state: np.random.RandomState
):
    """Test whether evaluating a random process returns a random variable."""
    n_inputs_x0 = 10
    x0 = random_state.normal(size=(n_inputs_x0, random_process.input_dim))
    y0 = random_process(x0)

    assert isinstance(y0, randvars.RandomVariable), (
        f"Output of {repr(random_process)} is not a " f"random variable."
    )


def test_samples_are_callables(
    random_process: randprocs.RandomProcess, random_state: np.random.RandomState
):
    """When not specifying inputs to the sample method it should return ``size`` number
    of callables."""
    assert callable(random_process.sample(random_state=random_state))


def test_rp_mean_cov_evaluated_matches_rv_mean_cov(
    random_process: randprocs.RandomProcess, random_state: np.random.RandomState
):
    """Check whether the evaluated mean and covariance function of a random process is
    equivalent to the mean and covariance of the evaluated random process as a random
    variable."""
    x = random_state.normal(size=(10, random_process.input_dim))

    np.testing.assert_allclose(
        random_process(x).mean,
        random_process.mean(x),
        err_msg=f"Mean of evaluated {repr(random_process)} does not match the "
        f"random process mean function evaluated.",
    )

    np.testing.assert_allclose(
        random_process(x).cov,
        random_process.cov(x),
        err_msg=f"Covariance of evaluated {repr(random_process)} does not match the "
        f"random process mean function evaluated.",
    )
