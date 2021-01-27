"""Tests for the uncertainty calibration procedure."""
import numpy as np
import pytest

from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.linalg.solvers.data import (
    LinearSolverAction,
    LinearSolverData,
    LinearSolverObservation,
)
from probnum.linalg.solvers.hyperparam_optim import UncertaintyCalibration
from probnum.problems import LinearSystem

# Filter warnings caused by GPy
pytestmark = [
    pytest.mark.filterwarnings("ignore:::GPy[.*]"),
    pytest.mark.filterwarnings("ignore:::paramz[.*]"),
]


def test_uncertainty_scales_are_inverses_of_each_other(
    uncertainty_calibration: UncertaintyCalibration,
    linsys_spd: LinearSystem,
    prior: LinearSystemBelief,
    solver_data: LinearSolverData,
):
    """Test whether any uncertainty calibration routine returns a pair of numbers which
    are inverses to each other."""
    unc_scales, _ = uncertainty_calibration(
        problem=linsys_spd,
        belief=prior,
        data=solver_data,
        solver_state=None,
    )
    np.testing.assert_approx_equal(
        unc_scales[0],
        1 / unc_scales[1],
        err_msg="Uncertainty scales for A and Ainv are not inverse to each other.",
    )


def test_calibration_after_one_iteration_returns_rayleigh_quotient(
    uncertainty_calibration: UncertaintyCalibration,
    linsys_spd: LinearSystem,
    prior: LinearSystemBelief,
    action: LinearSolverAction,
    matvec_observation: LinearSolverObservation,
):
    """Test whether calibrating for one action and observation returns the Rayleigh
    quotient as the uncertainty scale for A."""
    rayleigh_quotient = np.exp(
        np.log(action.actA.T @ matvec_observation.obsA)
        - np.log(action.actA.T @ action.actA)
    ).item()

    unc_scales, _ = uncertainty_calibration(
        problem=linsys_spd,
        belief=prior,
        data=LinearSolverData(
            actions=[action],
            observations=[matvec_observation],
        ),
        solver_state=None,
    )
    np.testing.assert_approx_equal(rayleigh_quotient, unc_scales[0])


def test_unknown_calibration_procedure():
    """Test whether an unknown calibration procedure raises a ValueError."""
    with pytest.raises(ValueError):
        UncertaintyCalibration(method="non-existent")
