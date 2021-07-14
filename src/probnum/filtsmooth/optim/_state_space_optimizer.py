"""Gauss-Newton methods in state-space models."""

import abc

from probnum.filtsmooth.optim import _stoppingcriterion


class StateSpaceOptimizer(abc.ABC):
    def __init__(self, kalman, stopping_criterion=None):
        self.kalman = kalman
        if stopping_criterion is None:
            stopping_criterion = _stoppingcriterion.StoppingCriterion()
        self.stopping_criterion = stopping_criterion

    def solve(self, regression_problem, initial_guess):

        # Initialization (in case the loop is empty)
        posterior = initial_guess
        info_dicts = None

        # Iterate right to the end of the generator
        solution_generator = self.solution_generator(
            regression_problem=regression_problem, initial_guess=initial_guess
        )
        for posterior, info_dicts in solution_generator:
            pass

        return posterior, info_dicts

    @abc.abstractmethod
    def solution_generator(self, regression_problem, initial_guess):
        raise NotImplementedError
