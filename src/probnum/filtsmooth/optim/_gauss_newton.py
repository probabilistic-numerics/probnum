"""Gauss-Newton methods in state-space models."""
from typing import Callable

import numpy as np

from probnum import problems, statespace

from ._state_space_optimizer import StateSpaceOptimizer

LinearizationStrategyType = Callable[
    [statespace.DiscreteGaussian], statespace.DiscreteLinearGaussian
]


class GaussNewton(StateSpaceOptimizer):
    """Gauss-Newton optimizer in state-space models.

    Equivalent to the iterated Kalman smoother.
    """

    def __init__(
        self,
        linearization_strategy: LinearizationStrategyType,
        kalman,
        stopping_criterion=None,
    ):
        super().__init__(kalman=kalman, stopping_criterion=stopping_criterion)

        self.linearization_strategy = linearization_strategy

        # The heavy lifting happens with an underlying Kalman filter.
        self.kalman = kalman

    def from_local_linearization(self, kalman, stopping_criterion=None):
        pass

    def solution_generator(self, regression_problem, initial_guess):

        new_posterior = initial_guess
        info_dicts = []
        yield new_posterior, info_dicts

        new_mean = new_posterior.states.mean
        old_mean = np.inf * np.ones(new_mean.shape)
        mean_difference = new_mean - old_mean

        while not self.stoppping_criterion.terminate(
            error=mean_difference, reference=new_mean
        ):

            old_posterior = new_posterior
            regression_problem = self._linearize_regression_problem(
                regression_problem=regression_problem, posterior=new_posterior
            )
            new_posterior, info_dicts = self._kalman.filtsmooth(
                regression_problem,
            )
            yield new_posterior, info_dicts
            new_mean = new_posterior.states.mean
            old_mean = old_posterior.states.mean
            mean_difference = new_mean - old_mean

    def _linearize_regression_problem(self, regression_problem, posterior):
        new_measmod_list = self._linearize_measurement_model_list(
            measurement_model_list=regression_problem.measurement_models,
            states=posterior.states,
        )
        return problems.TimeSeriesRegressionProblem(
            locations=regression_problem.locations,
            measurement_models=new_measmod_list,
            observations=regression_problem.observations,
            solution=regression_problem.solution,
        )

    def _linearize_measurement_model_list(self, measurement_model_list, states):
        new_measmod_list = [
            self.linearization_strategy(mm, x)
            for mm, x in zip(measurement_model_list, states)
        ]
        return new_measmod_list
