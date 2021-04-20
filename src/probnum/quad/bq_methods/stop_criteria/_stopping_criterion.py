"""Stopping criteria for Bayesian quadrature."""

from typing import Callable, Optional

from probnum.type import FloatArgType, IntArgType


class StoppingCriterion:
    def __init__(self, stopping_criterion: Callable):
        self._stopping_criterion = stopping_criterion

    def __call__(self, integral_belief, bq_state) -> bool:
        return self._stopping_criterion(bq_state)


class IntegralVariance(StoppingCriterion):
    def __init__(self, var_tol: FloatArgType = None):
        self.var_tol = var_tol
        super().__init__(stopping_criterion=self.__call__)

    def __call__(self, integral_belief, bq_state) -> bool:
        return integral_belief.var <= self.var_tol


class RelativeError(StoppingCriterion):
    def __init__(self, rel_tol: FloatArgType = None):
        self.rel_tol = rel_tol
        super().__init__(stopping_criterion=self.__call__())

    def __call__(self, integral_belief, bq_state) -> bool:
        # TODO: IMPLEMENT THIS! WILL REQUIRE ALSO THE PREVIOUS BQ_STATE!
        return True


class MaxNevals(StoppingCriterion):
    def __init__(self, max_nevals: IntArgType = None):
        self.max_nevals = max_nevals
        super().__init__(stopping_criterion=self.__call__)

    def __call__(self, integral_belief, bq_state) -> bool:
        return bq_state.info.nevals >= self.max_nevals
