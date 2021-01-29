"""Test fixtures for hyperparameter optimization methods."""

import pytest

from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.linalg.solvers.data import LinearSolverData
from probnum.linalg.solvers.hyperparam_optim import (
    OptimalNoiseScale,
    UncertaintyCalibration,
)
from probnum.problems import NoisyLinearSystem


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
) -> UncertaintyCalibration:
    """Uncertainty calibration method for probabilistic linear solvers."""
    return UncertaintyCalibration(method=calibration_method)


@pytest.fixture(
    params=[
        pytest.param(iterative, id=iterative[0])
        for iterative in [("iter", True), ("batch", False)]
    ],
    name="optimal_noise_scale",
)
def fixture_optimal_noise_scale(
    request,
    eps: float,
    linsys_matnoise: NoisyLinearSystem,
    prior: LinearSystemBelief,
    noisy_solver_data: LinearSolverData,
) -> float:
    """Computes the optimal noise scale of a noisy linear system."""
    noiseA = OptimalNoiseScale(iterative=request.param[1])(
        problem=linsys_matnoise, belief=prior, data=noisy_solver_data
    )
    return noiseA.epsA_cov.A.args[1]
