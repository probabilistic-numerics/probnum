"""Test cases for hyperparameter optimization of probabilistic linear solvers."""

from typing import Union

import numpy as np

from probnum.linalg.linearsolvers.hyperparam_optim import (
    OptimalNoiseScale,
    UncertaintyCalibration,
)
from probnum.problems import LinearSystem
from tests.testing import NumpyAssertions

from .test_probabilistic_linear_solver import ProbabilisticLinearSolverTestCase

# pylint: disable="invalid-name"


class HyperparameterOptimizationTestCase(
    ProbabilisticLinearSolverTestCase, NumpyAssertions
):
    """General test case for hyperparameter optimization of probabilistic linear
    solvers."""

    def setUp(self) -> None:
        """Test resources for linear solver hyperparameter optimization."""
        self.iteration = 5
        self.actions = [
            col[:, None]
            for col in self.rng.normal(size=(self.linsys.A.shape[0], self.iteration)).T
        ]
        self.observations = [self.linsys.A @ s for s in self.actions]

        self.hyperparam_optim_routines = [
            UncertaintyCalibration(method="gpkern"),
            OptimalNoiseScale(),
        ]

    def test_returns_hyperparameter_tuple(self):
        """Test whether a hyperparameter optimization procedures returns a tuple of
        optimized hyperparameters."""


class UncertaintyCalibrationTestCase(HyperparameterOptimizationTestCase):
    """Test case for the uncertainty calibration procedure."""

    def setUp(self) -> None:
        """Test resources for uncertainty calibration."""
        self.iteration = 5
        self.actions = [
            col[:, None]
            for col in self.rng.normal(size=(self.linsys.A.shape[0], self.iteration)).T
        ]
        self.observations = [self.linsys.A @ s for s in self.actions]

        self.uncertainty_calibration_routines = [
            UncertaintyCalibration(method="adhoc"),
            UncertaintyCalibration(method="weightedmean"),
            UncertaintyCalibration(method="gpkern"),
        ]

    def test_uncertainty_scales_are_inverses_of_each_other(self):
        """Test whether any uncertainty calibration routine returns a pair of numbers
        which are inverses to each other."""
        for calib_routine in self.uncertainty_calibration_routines:
            with self.subTest():
                unc_scales, _, _ = calib_routine(
                    problem=self.linsys,
                    belief=self.prior,
                    actions=self.actions,
                    observations=self.observations,
                    solver_state=None,
                )
                self.assertApproxEqual(
                    unc_scales[0],
                    1 / unc_scales[1],
                    msg="Uncertainty scales for A and Ainv are not "
                    "inverse to each other.",
                )

    def test_calibration_after_one_iteration_returns_rayleigh_quotient(self):
        """Test whether calibrating for one action and observation returns the Rayleigh
        quotient as the uncertainty scale for A."""
        rayleigh_quotient = np.exp(
            np.log(self.actions[0].T @ self.observations[0])
            - np.log(self.actions[0].T @ self.actions[0])
        ).item()

        for calib_routine in self.uncertainty_calibration_routines:
            with self.subTest():
                unc_scales, _, _ = calib_routine(
                    problem=self.linsys,
                    belief=self.prior,
                    actions=[self.actions[0]],
                    observations=[self.observations[0]],
                    solver_state=None,
                )
            self.assertApproxEqual(rayleigh_quotient, unc_scales[0])

    def test_unknown_calibration_procedure(self):
        """Test whether an unknown calibration procedure raises a ValueError."""
        with self.assertRaises(ValueError):
            UncertaintyCalibration(method="non-existent")(
                problem=self.linsys,
                belief=self.prior,
                actions=self.actions,
                observations=self.observations,
                solver_state=None,
            )


class OptimalNoiseScaleTestCase(HyperparameterOptimizationTestCase):
    """Test case for the optimization of the noise scale."""

    def setUp(self) -> None:
        """Test resources for noise scale optimization."""
        self.iteration = 5
        self.actions = [
            col[:, None]
            for col in self.rng.normal(size=(self.linsys.A.shape[0], self.iteration)).T
        ]
        self.observations = [self.linsys.A @ s for s in self.actions]

    def test_iterative_and_batch_identical(self):
        """Test whether computing the optimal scale for k observations in a batch
        matches the recursive form of the optimal noise scale."""

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
