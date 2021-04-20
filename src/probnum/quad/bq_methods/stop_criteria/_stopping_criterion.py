"""Stopping criteria for Bayesian quadrature."""

from typing import Callable, Optional

from probnum.type import FloatArgType, IntArgType


class StoppingCriterion:
    def __init__(self, stopping_criterion: Callable):
        self._stopping_criterion = stopping_criterion

    def __call__(self, integral_belief, bq_state) -> bool:
        return self._stopping_criterion(bq_state)


class IntegralVariance(StoppingCriterion):
    def __init__(self, variance_tol: FloatArgType = None):
        self.variance_tol = variance_tol
        super().__init__(stopping_criterion=self.__call__)

    def __call__(self, integral_belief, bq_state) -> bool:
        if self.variance_tol is None:
            _variance_tol = 1e-3
        else:
            _variance_tol = self.variance_tol
        return integral_belief.var <= _variance_tol


class MaxIterations(StoppingCriterion):
    def __init__(self, maxiter: IntArgType = None):
        self.maxiter = maxiter
        super().__init__(stopping_criterion=self.__call__)

    def __call__(self, integral_belief, bq_state) -> bool:
        if self.maxiter is None:
            _maxiter = bq_state.dim * 20
        else:
            _maxiter = self.maxiter
        print(_maxiter, bq_state.info.iteration)
        return bq_state.info.iteration >= _maxiter
