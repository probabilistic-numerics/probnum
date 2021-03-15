"""Tests for diffusions and calibration."""

import abc

import numpy as np
import pytest

from probnum import diffeq, random_variables


@pytest.fixture
def some_meas_rv():
    """Generic measurement RV used to test calibration.

    This config should return 9.776307498421126 for
    Diffusion.calibrate_locally.
    """
    some_mean = np.arange(10, 13)
    some_cov = np.arange(9).reshape((3, 3)) @ np.arange(9).reshape((3, 3)).T + np.eye(3)
    meas_rv = random_variables.Normal(some_mean, some_cov)
    return meas_rv


class DiffusionTestInterface(abc.ABC):
    @abc.abstractmethod
    def test_call(self):
        raise NotImplementedError

    def test_calibrate_locally(self, some_meas_rv):
        # 9.776307498421126 is the true value for given some_meas_rv
        out = self.diffusion.calibrate_locally(some_meas_rv)
        np.testing.assert_allclose(out, 9.776307498421126)

    @abc.abstractmethod
    def test_calibrate_all_states(self):
        raise NotImplementedError


class TestConstantDiffusion(DiffusionTestInterface):

    all_diffusion_returns = pytest.mark.parametrize(
        "use_global_estimate_as_local_estimate", [True, False]
    )

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, use_global_estimate_as_local_estimate):
        self.diffusion = diffeq.ConstantDiffusion(
            use_global_estimate_as_local_estimate=use_global_estimate_as_local_estimate
        )

    @all_diffusion_returns
    def test_call(self, use_global_estimate_as_local_estimate):
        generic_diffusion_value = 1.2345
        self.diffusion.update_current_information(
            generic_diffusion_value, generic_diffusion_value, 0.0
        )
        out = self.diffusion(0.5)
        np.testing.assert_allclose(out, generic_diffusion_value)

    @all_diffusion_returns
    def test_calibrate_locally(
        self, some_meas_rv, use_global_estimate_as_local_estimate
    ):
        # 9.776307498421126 is the true value for given some_meas_rv
        out = self.diffusion.calibrate_locally(some_meas_rv)
        np.testing.assert_allclose(out, 9.776307498421126)

    @all_diffusion_returns
    def test_update_current_information(self, use_global_estimate_as_local_estimate):
        diffusion_list = np.arange(100, 110)
        for diff in diffusion_list:
            # t = None does not make a difference
            self.diffusion.update_current_information(diff, diff, None)
        np.testing.assert_allclose(
            self.diffusion.diffusion, self.diffusion.diffusion, diffusion_list.mean()
        )

    @all_diffusion_returns
    def test_calibrate_all_states(
        self, some_meas_rv, use_global_estimate_as_local_estimate
    ):

        # Set up 10 RVs
        meas_rvs = [some_meas_rv for _ in range(10)]

        # Compute 10 diffusions
        diffusion_list = np.arange(100, 110)
        for diff in diffusion_list:
            # t = None does not make a difference
            self.diffusion.update_current_information(diff, diff, None)

        calibrated_states = self.diffusion.calibrate_all_states(
            meas_rvs, locations=None
        )
        for calibrated, uncalibrated in zip(calibrated_states, meas_rvs):
            np.testing.assert_allclose(
                calibrated.cov, self.diffusion.diffusion * uncalibrated.cov
            )


class TestPiecewiseConstantDiffusion(DiffusionTestInterface):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.diffusion = diffeq.PiecewiseConstantDiffusion()

    def test_call(self):

        # Compute 10 diffusions
        times = np.arange(10)
        diffusion_list = 1 + 0.1 * np.random.rand(10)
        for diff, t in zip(diffusion_list, times):
            self.diffusion.update_current_information(diff, diff, t)

        for diff, t in zip(diffusion_list, times):

            # Diffusion step functions are right-continuous,
            # hence we move to the left to get the expected output
            # Shift is in (0.1, 0.2) and we randomize it
            # to make the test stricter
            t_eval = t - 0.1 * (1 + np.random.rand())
            received = self.diffusion(t_eval)
            np.testing.assert_almost_equal(received, diff)

        # Test the extrapolation diffusion, too.
        # Only extrapolation to the right, which should be the final element in the list.
        # Extrapolation to the left is covered in the first iteration of the loop above.
        t_eval = self.diffusion.locations[-1] + 1.2345
        received = self.diffusion(t_eval)
        np.testing.assert_almost_equal(received, self.diffusion.diffusions[-1])

    def test_calibrate_all_states(self, some_meas_rv):

        # Set up 10 RVs
        meas_rvs = [some_meas_rv for _ in range(10)]

        # Compute 10 diffusions
        diffusion_list = np.arange(100, 110)
        for diff in diffusion_list:
            # t = None does not make a difference
            self.diffusion.update_current_information(diff, diff, None)

        calibrated_states = self.diffusion.calibrate_all_states(
            meas_rvs, locations=None
        )

        # Assert that nothing happens.
        for calibrated, uncalibrated in zip(calibrated_states, meas_rvs):
            np.testing.assert_allclose(calibrated.cov, uncalibrated.cov)
