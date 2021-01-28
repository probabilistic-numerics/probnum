"""Test fixtures for hyperparameter optimization methods."""

import pytest

from probnum.linalg.solvers import hyperparam_optim


@pytest.fixture(
    params=[
        pytest.param(calibration_method, id=calibration_method)
        for calibration_method in ["adhoc", "weightedmean", "gpkern"]
    ],
    name="calibration_method",
)
def fixture_calibration_method(request) -> str:
    """Names of available uncertainty calibration methods."""
    return request.param


@pytest.fixture(name="uncertainty_calibration")
def fixture_uncertainty_calibration(
    calibration_method: str,
) -> hyperparam_optim.UncertaintyCalibration:
    """Uncertainty calibration method for probabilistic linear solvers."""
    return hyperparam_optim.UncertaintyCalibration(method=calibration_method)


@pytest.fixture(
    params=[
        pytest.param(iterative, id=iterative[0])
        for iterative in [("iter", True), ("batch", False)]
    ],
    name="optimal_noise_scale",
)
def fixture_optimal_noise_scale(request) -> hyperparam_optim.OptimalNoiseScale:
    """Computes the optimal noise scale of a noisy linear system."""
    return hyperparam_optim.OptimalNoiseScale(iterative=request.param[1])
