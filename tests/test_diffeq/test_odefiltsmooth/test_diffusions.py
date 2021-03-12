"""Tests for diffusions and calibration."""

import numpy as np
import pytest

from probnum import diffeq, random_variables

##########################
# Constant diffusion tests
##########################


@pytest.fixture
def constant_diffusion():
    constant_diffusion = diffeq.ConstantDiffusion()
    constant_diffusion.update_current_information(diffusion=1.0, t=0.0)
    return constant_diffusion


def test_call(constant_diffusion):
    out = constant_diffusion(0.5)
    np.testing.assert_allclose(out, 1.0)


@pytest.mark.parametrize("idx", [0, 1, 100])
def test_getitem(constant_diffusion, idx):
    out = constant_diffusion[0]
    np.testing.assert_allclose(out, 1.0)


def test_calibrate_locally(constant_diffusion):

    # This config should return 9.776307498421126
    some_mean = np.arange(10, 13)
    some_cov = np.arange(9).reshape((3, 3)) @ np.arange(9).reshape((3, 3)).T + np.eye(3)
    meas_rv = random_variables.Normal(some_mean, some_cov)

    out = constant_diffusion.calibrate_locally(meas_rv)

    np.testing.assert_allclose(out, 9.776307498421126)


def test_update_current_information():
    constant_diffusion = diffeq.ConstantDiffusion()
    diffusion_list = np.arange(100, 110)
    for diff in diffusion_list:
        # t = None does not make a difference
        constant_diffusion.update_current_information(diff, None)

    np.testing.assert_allclose(constant_diffusion.diffusion, diffusion_list.mean())


def test_postprocess_states(constant_diffusion):

    # Set up 10 RVs
    some_mean = np.arange(10, 13)
    some_cov = np.arange(9).reshape((3, 3)) @ np.arange(9).reshape((3, 3)).T + np.eye(3)
    meas_rvs = [random_variables.Normal(some_mean, some_cov) for _ in range(10)]

    # Compute 10 diffusions
    diffusion_list = np.arange(100, 110)
    for diff in diffusion_list:
        # t = None does not make a difference
        constant_diffusion.update_current_information(diff, None)

    calibrated_states = constant_diffusion.postprocess_states(meas_rvs)
    for calibrated, uncalibrated in zip(calibrated_states, meas_rvs):
        np.testing.assert_allclose(
            calibrated.cov, constant_diffusion.diffusion * uncalibrated.cov
        )
