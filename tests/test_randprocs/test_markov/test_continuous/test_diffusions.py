"""Tests for diffusions and calibration."""

import abc

import numpy as np
import pytest

from probnum import randprocs, randvars


@pytest.fixture
def some_meas_rv1():
    """Generic measurement RV used to test calibration.

    This config should return 9.776307498421126 for
    Diffusion.calibrate_locally.
    """
    some_mean = np.arange(10, 13)
    some_cov = np.arange(9).reshape((3, 3)) @ np.arange(9).reshape((3, 3)).T + np.eye(3)
    meas_rv = randvars.Normal(some_mean, some_cov)
    return meas_rv


@pytest.fixture
def some_meas_rv2():
    """Another generic measurement RV used to test calibration.

    This config should return 9.776307498421126 for
    Diffusion.calibrate_locally.
    """
    some_mean = np.arange(10, 13)
    some_cov = np.arange(3, 12).reshape((3, 3)) @ np.arange(3, 12).reshape(
        (3, 3)
    ).T + np.eye(3)
    meas_rv = randvars.Normal(some_mean, some_cov)
    return meas_rv


@pytest.fixture
def t():
    return 1.234567654231


class DiffusionTestInterface(abc.ABC):
    @abc.abstractmethod
    def test_call(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test_estimate_locally_and_update_in_place(
        self, some_meas_rv1, some_meas_rv2, t
    ):
        raise NotImplementedError


class TestConstantDiffusion(DiffusionTestInterface):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.diffusion = randprocs.markov.continuous.ConstantDiffusion()

    def test_call(self):
        """Check whether call returns the correct values."""
        unimportant_time_value = 0.5
        with pytest.raises(NotImplementedError):
            self.diffusion(unimportant_time_value)

        self.diffusion.diffusion = 3.45678
        assert self.diffusion(unimportant_time_value) == 3.45678

    def test_estimate_locally_and_update_in_place(
        self, some_meas_rv1, some_meas_rv2, t
    ):
        sigma_squared = self.diffusion.estimate_locally(
            meas_rv=some_meas_rv1, meas_rv_assuming_zero_previous_cov=some_meas_rv2, t=t
        )
        self.diffusion.update_in_place(sigma_squared, t=t)
        expected = (
            some_meas_rv1.mean
            @ np.linalg.solve(some_meas_rv1.cov, some_meas_rv1.mean)
            / some_meas_rv1.size
        )
        np.testing.assert_allclose(sigma_squared, expected)

        unimportant_time_value = 0.1234578
        np.testing.assert_allclose(self.diffusion(unimportant_time_value), expected)


class TestPiecewiseConstantDiffusion(DiffusionTestInterface):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.diffusion = randprocs.markov.continuous.PiecewiseConstantDiffusion(t0=0.0)

    def test_call(self):

        unimportant_time_value = 0.5
        with pytest.raises(NotImplementedError):
            self.diffusion(unimportant_time_value)

        diffusions = np.random.rand(10)
        self.diffusion._diffusions = diffusions

        self.diffusion._locations = np.arange(1.0, 11.0, step=1.0)

        times = [1.5, -1.0, 100.0, 11.0]
        expected_values = [diffusions[0], diffusions[0], diffusions[-1], diffusions[-1]]
        for t, val in zip(times, expected_values):
            received = self.diffusion(t)
            print(t)
            np.testing.assert_allclose(received, val)

        # Again, but vectorised
        np.testing.assert_allclose(self.diffusion(times), expected_values)

    def test_estimate_locally_and_update_in_place(
        self, some_meas_rv1, some_meas_rv2, t
    ):
        sigma_squared = self.diffusion.estimate_locally(
            meas_rv=some_meas_rv1, meas_rv_assuming_zero_previous_cov=some_meas_rv2, t=t
        )
        self.diffusion.update_in_place(sigma_squared, t=t)

        expected = (
            some_meas_rv2.mean
            @ np.linalg.solve(some_meas_rv2.cov, some_meas_rv2.mean)
            / some_meas_rv2.size
        )
        np.testing.assert_allclose(sigma_squared, expected)

        unimportant_time_value = 0.1234578
        np.testing.assert_allclose(self.diffusion(unimportant_time_value), expected)
