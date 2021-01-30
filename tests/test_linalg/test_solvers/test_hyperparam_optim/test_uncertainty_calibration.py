"""Tests for the uncertainty calibration procedure."""
import numpy as np
import pytest

from probnum.linalg.solvers import (
    ProbabilisticLinearSolver,
    beliefs,
    observation_ops,
    policies,
    stop_criteria,
)
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
    unc_scales = uncertainty_calibration(
        problem=linsys_spd,
        belief=prior,
        data=solver_data,
        solver_state=None,
    )
    np.testing.assert_approx_equal(
        unc_scales.Phi,
        1 / unc_scales.Psi,
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

    unc_scales = uncertainty_calibration(
        problem=linsys_spd,
        belief=prior,
        data=LinearSolverData(
            actions=[action],
            observations=[matvec_observation],
        ),
        solver_state=None,
    )
    np.testing.assert_approx_equal(rayleigh_quotient, unc_scales.Phi)


def test_unknown_calibration_procedure():
    """Test whether an unknown calibration procedure raises a ValueError."""
    with pytest.raises(ValueError):
        UncertaintyCalibration(method="non-existent")


def test_uncertainty_calibration_error(
    linsys_spd: LinearSystem, uncertainty_calibration: UncertaintyCalibration
):
    """Test if the available uncertainty calibration procedures affect the error of the
    returned solution."""
    pls = ProbabilisticLinearSolver(
        prior=beliefs.WeakMeanCorrespondenceBelief.from_scalar(
            scalar=1.0,
            problem=linsys_spd,
        ),
        hyperparam_optim_method=uncertainty_calibration,
        policy=policies.ConjugateDirections(),
        observation_op=observation_ops.MatVec(),
        stopping_criteria=[stop_criteria.MaxIterations(), stop_criteria.Residual()],
    )

    belief, solver_state = pls.solve(linsys_spd)
    xdiff = linsys_spd.solution - belief.x.mean

    assert (xdiff.T @ linsys_spd.A @ xdiff).item() == pytest.approx(0.0, abs=10 ** -6)
