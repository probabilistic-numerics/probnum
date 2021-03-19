"""Tests for random processes."""

import numpy as np
import pytest

from probnum import kernels, randprocs


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


def test_samples_are_callables(
    random_process: randprocs.RandomProcess, random_state: np.random.RandomState
):
    """When not specifying inputs to the sample method it should return ``size`` number
    of callables."""
    assert callable(random_process.sample(random_state=random_state))


def test_rp_mean_cov_evaluated_matches_rv_mean_cov(
    random_process: randprocs.RandomProcess,
):
    """Check whether the evaluated mean and covariance function of a random process is
    equivalent to the mean and covariance of the evaluated random process as a random
    variable."""
    x = np.random.normal(size=(10, random_process.input_dim))

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
