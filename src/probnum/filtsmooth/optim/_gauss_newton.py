"""Gauss-Newton methods in state-space models."""
from typing import Callable

import numpy as np

from probnum import problems, statespace

from ._state_space_optimizer import StateSpaceOptimizer


class GaussNewton(StateSpaceOptimizer):
    """Gauss-Newton optimizer in state-space models.

    Equivalent to the iterated Kalman smoother.
    """

    def solution_generator(self, regression_problem, initial_guess):

        new_posterior = initial_guess
        info_dicts = []
        yield new_posterior, info_dicts

        new_mean = new_posterior.states.mean
        old_mean = np.inf * np.ones(new_mean.shape)
        mean_difference = new_mean - old_mean

        while not self.stopping_criterion.terminate(
            error=mean_difference, reference=new_mean
        ):

            old_posterior = new_posterior
            new_posterior, info_dicts = self.kalman.filtsmooth(
                regression_problem, _previous_posterior=old_posterior
            )
            yield new_posterior, info_dicts
            new_mean = new_posterior.states.mean
            old_mean = old_posterior.states.mean
            mean_difference = new_mean - old_mean
