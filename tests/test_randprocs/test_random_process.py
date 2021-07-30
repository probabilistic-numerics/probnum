"""Tests for random processes."""

import numpy as np
import pytest

from probnum import randprocs, randvars

# pylint: disable=invalid-name


def test_output_shape(random_process: randprocs.RandomProcess, args0: np.ndarray):
    """Test whether evaluations of the random process have the correct shape."""
    if random_process.output_dim == 1:
        assert random_process(args0).ndim == 1
    else:
        assert random_process(args0).shape[1] == random_process.output_dim


def test_mean_shape(random_process: randprocs.RandomProcess, args0: np.ndarray):
    """Test whether the mean of the random process has the correct shape."""
    if random_process.output_dim == 1:
        assert random_process.mean(args0).ndim == 1
    else:
        assert random_process.mean(args0).shape[1] == random_process.output_dim


def test_var_shape(random_process: randprocs.RandomProcess, args0: np.ndarray):
    """Test whether the variance of the random process has the correct shape."""
    if random_process.output_dim == 1:
        assert random_process.var(args0).ndim == 1
    else:
        assert random_process.var(args0).shape[1] == random_process.output_dim


def test_std_shape(random_process: randprocs.RandomProcess, args0: np.ndarray):
    """Test whether the standard deviation of the random process has the correct
    shape."""
    if random_process.output_dim == 1:
        assert random_process.std(args0).ndim == 1
    else:
        assert random_process.std(args0).shape[1] == random_process.output_dim


def test_cov_shape(random_process: randprocs.RandomProcess, args0: np.ndarray):
    """Test whether the covariance of the random process has the correct shape."""
    n = args0.shape[0]
    if random_process.output_dim == 1:
        assert (
            random_process.cov(args0).shape == (n, n)
            or random_process.cov(args0).ndim < 2
        )
    else:
        assert random_process.cov(args0).shape == (
            n,
            n,
            random_process.output_dim,
            random_process.output_dim,
        )


def test_evaluated_random_process_is_random_variable(
    random_process: randprocs.RandomProcess, rng: np.random.Generator
):
    """Test whether evaluating a random process returns a random variable."""
    n_inputs_args0 = 10
    args0 = rng.normal(size=(n_inputs_args0, random_process.input_dim))
    y0 = random_process(args0)

    assert isinstance(y0, randvars.RandomVariable), (
        f"Output of {repr(random_process)} is not a " f"random variable."
    )


@pytest.mark.xfail(reason="Not yet implemented for random processes.")
def test_samples_are_callables(
    random_process: randprocs.RandomProcess, rng: np.random.Generator
):
    """When not specifying inputs to the sample method it should return ``size`` number
    of callables."""
    assert callable(random_process.sample(rng=rng))


@pytest.mark.xfail(reason="Not yet implemented for random processes.")
def test_sample_paths_are_deterministic_functions(
    random_process: randprocs.RandomProcess, args0: np.ndarray
):
    """When sampling paths from a random process, repeated evaluation of the sample path
    at the same inputs should return the same values."""
    sample_path = random_process.sample()
    np.testing.assert_array_equal(sample_path(args0), sample_path(args0))


def test_rp_mean_cov_evaluated_matches_rv_mean_cov(
    random_process: randprocs.RandomProcess, rng: np.random.Generator
):
    """Check whether the evaluated mean and covariance function of a random process is
    equivalent to the mean and covariance of the evaluated random process as a random
    variable."""
    x = rng.normal(size=(10, random_process.input_dim))

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
