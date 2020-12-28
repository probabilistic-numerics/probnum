"""Test cases for hyperparameter optimization of probabilistic linear solvers."""
from typing import Optional

import numpy as np

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import LinearSolverState, LinearSystemBelief
from probnum.linalg.linearsolvers.hyperparam_optim import (
    HyperparameterOptimization,
    OptimalNoiseScale,
    UncertaintyCalibration,
)
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix
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
        self.actions = self.rng.normal(size=(self.linsys.A.shape[0], 3))
        self.observations = self.linsys.A @ self.actions

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


class OptimalNoiseScaleTestCase(HyperparameterOptimizationTestCase):
    """Test case for the optimization of the noise scale."""

    def setUp(self) -> None:
        """Test resources for noise scale optimization."""
        self.iteration = 5
        self.actions = self.rng.normal(size=(self.linsys.A.shape[0], self.iteration))
        self.observations = self.linsys.A @ self.actions + self.rng.normal(
            size=(self.linsys.A.shape[0], self.iteration)
        )
