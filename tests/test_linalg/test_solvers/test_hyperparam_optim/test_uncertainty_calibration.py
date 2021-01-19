"""Tests for the uncertainty calibration procedure."""
import numpy as np
import pytest

from probnum.linalg.solvers.beliefs import LinearSystemBelief
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
    actions: list,
    matvec_observations: list,
):
    """Test whether any uncertainty calibration routine returns a pair of numbers which
    are inverses to each other."""
    unc_scales, _ = uncertainty_calibration(
        problem=linsys_spd,
        belief=prior,
        actions=actions,
        observations=matvec_observations,
        solver_state=None,
    )
    np.testing.assert_approx_equal(
        unc_scales[0],
        1 / unc_scales[1],
        err_msg="Uncertainty scales for A and Ainv are not " "inverse to each other.",
    )


def test_calibration_after_one_iteration_returns_rayleigh_quotient(
    uncertainty_calibration: UncertaintyCalibration,
    linsys_spd: LinearSystem,
    prior: LinearSystemBelief,
    action: np.ndarray,
    matvec_observation: np.ndarray,
):
    """Test whether calibrating for one action and observation returns the Rayleigh
    quotient as the uncertainty scale for A."""
    rayleigh_quotient = np.exp(
        np.log(action.T @ matvec_observation) - np.log(action.T @ action)
    ).item()

    unc_scales, _ = uncertainty_calibration(
        problem=linsys_spd,
        belief=prior,
        actions=[action],
        observations=[matvec_observation],
        solver_state=None,
    )
    np.testing.assert_approx_equal(rayleigh_quotient, unc_scales[0])


def test_unknown_calibration_procedure():
    """Test whether an unknown calibration procedure raises a ValueError."""
    with pytest.raises(ValueError):
        UncertaintyCalibration(method="non-existent")
