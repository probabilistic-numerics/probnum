"""Test cases for hyperparameter optimization of probabilistic linear solvers."""

import numpy as np
import pytest

from probnum.linalg.linearsolvers.beliefs import LinearSystemBelief
from probnum.linalg.linearsolvers.hyperparam_optim import (
    OptimalNoiseScale,
    UncertaintyCalibration,
)
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"


def test_returns_hyperparameter_tuple():
    """Test whether a hyperparameter optimization procedures returns a tuple of
    optimized hyperparameters."""
    # TODO


class TestUncertaintyCalibration:
    """Tests for the uncertainty calibration procedure."""

    def test_uncertainty_scales_are_inverses_of_each_other(
        self,
        uncertainty_calibration: UncertaintyCalibration,
        linsys_spd: LinearSystem,
        prior: LinearSystemBelief,
        actions: list,
        matvec_observations: list,
    ):
        """Test whether any uncertainty calibration routine returns a pair of numbers
        which are inverses to each other."""
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
            err_msg="Uncertainty scales for A and Ainv are not "
            "inverse to each other.",
        )

    def test_calibration_after_one_iteration_returns_rayleigh_quotient(
        self,
        uncertainty_calibration: UncertaintyCalibration,
        linsys_spd: LinearSystem,
        prior: LinearSystemBelief,
        action: np.ndarray,
        matvec_observation: np.ndarray,
    ):
        """Test whether calibrating for one action and observation returns the Rayleigh
        quotient as the uncertainty scale for A."""
        rayleigh_quotient = np.exp(
            np.log(action.T @ matvec_observation)
            - np.log(action.T @ matvec_observation)
        ).item()

        unc_scales, _ = uncertainty_calibration(
            problem=linsys_spd,
            belief=prior,
            actions=[action],
            observations=[matvec_observation],
            solver_state=None,
        )
        np.testing.assert_approx_equal(rayleigh_quotient, unc_scales[0])

    def test_unknown_calibration_procedure(self):
        """Test whether an unknown calibration procedure raises a ValueError."""
        with pytest.raises(ValueError):
            UncertaintyCalibration(method="non-existent")


class OptimalNoiseScaleTestCase:
    """Tests for the optimization of the noise scale."""

    def test_iterative_and_batch_identical(self):
        """Test whether computing the optimal scale for k observations in a batch
        matches the recursive form of the optimal noise scale."""
        # TODO
        # # Batch computed optimal noise scale
        # noise_scale_batch = OptimalNoiseScale._optimal_noise_scale_batch(
        #     problem=None,
        #     prior=None,
        #     actions=None,
        #     observations=None,
        # )
        #
        # # Iteratively computed optimal noise scale
        # noise_scale_iter = OptimalNoiseScale._optimal_noise_scale_iterative(
        #     problem=None,
        #     belief=None,
        #     action=None,
        #     observation=None,
        # )
