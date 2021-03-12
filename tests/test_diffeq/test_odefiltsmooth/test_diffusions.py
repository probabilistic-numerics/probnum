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
        pass

    @abc.abstractmethod
    def test_getitem(self):
        pass

    def test_calibrate_locally(self, some_meas_rv):
        # 9.776307498421126 is the true value for given some_meas_rv
        out = self.diffusion.calibrate_locally(some_meas_rv)
        np.testing.assert_allclose(out, 9.776307498421126)

    @abc.abstractmethod
    def test_update_current_information(self):
        pass

    @abc.abstractmethod
    def test_postprocess_states(self):
        pass


class TestConstantDiffusion(DiffusionTestInterface):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.diffusion = diffeq.ConstantDiffusion()

    def test_call(self):
        generic_diffusion_value = 1.2345
        self.diffusion.update_current_information(
            diffusion=generic_diffusion_value, t=0.0
        )
        out = self.diffusion(0.5)
        np.testing.assert_allclose(out, generic_diffusion_value)

    @pytest.mark.parametrize("idx", [0, 1, 100])
    def test_getitem(self, idx):
        generic_diffusion_value = 1.2345
        self.diffusion.update_current_information(
            diffusion=generic_diffusion_value, t=0.0
        )

        out = self.diffusion[0]
        np.testing.assert_allclose(out, generic_diffusion_value)

    def test_update_current_information(self):
        diffusion_list = np.arange(100, 110)
        for diff in diffusion_list:
            # t = None does not make a difference
            self.diffusion.update_current_information(diff, None)
        np.testing.assert_allclose(self.diffusion.diffusion, diffusion_list.mean())

    def test_postprocess_states(self, some_meas_rv):

        # Set up 10 RVs
        meas_rvs = [some_meas_rv for _ in range(10)]

        # Compute 10 diffusions
        diffusion_list = np.arange(100, 110)
        for diff in diffusion_list:
            # t = None does not make a difference
            self.diffusion.update_current_information(diff, None)

        calibrated_states = self.diffusion.postprocess_states(meas_rvs, locations=None)
        for calibrated, uncalibrated in zip(calibrated_states, meas_rvs):
            np.testing.assert_allclose(
                calibrated.cov, self.diffusion.diffusion * uncalibrated.cov
            )
