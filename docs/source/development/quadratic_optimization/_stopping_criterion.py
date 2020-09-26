from typing import Callable, Tuple, Union
from probnum.type import FloatArgType, IntArgType

import numpy as np
import probnum as pn


class QuadOptStoppingCriterion:
    """
    Stopping criterion for a 1D quadratic optimization problem.

    Parameters
    ----------
    stopping_criterion :
        Stopping criterion determining whether to stop or not.
    """

    def __init__(
        self,
        stopping_criterion: Callable[
            [Callable[[FloatArgType], FloatArgType], pn.RandomVariable, IntArgType],
            Tuple[bool, Union[str, None]],
        ],
    ):
        self._stopping_criterion = stopping_criterion

    def __call__(
        self,
        fun: Callable[[FloatArgType], FloatArgType],
        fun_params0: pn.RandomVariable,
        current_iter: IntArgType,
    ) -> Tuple[bool, Union[str, None]]:
        """
        Evaluate whether to stop.

        Parameters
        ----------
        fun :
            One-dimensional objective function.
        fun_params0 :
            Belief over the parameters of the quadratic objective.
        current_iter :
            Current iteration of the PN method.
        """
        return self._stopping_criterion(fun, fun_params0, current_iter)


class ParameterUncertainty(QuadOptStoppingCriterion):
    """
    Stopping criterion based on uncertainty about the parameters of the objective.

    Parameters
    ----------
    abstol :
        Absolute convergence tolerance.
    reltol :
        Relative convergence tolerance.
    """

    def __init__(self, abstol: FloatArgType, reltol: FloatArgType):
        self.abstol = abstol
        self.reltol = reltol
        super().__init__(
            stopping_criterion=self._parameter_uncertainty_stopping_criterion
        )

    def _parameter_uncertainty_stopping_criterion(
        self,
        fun: Callable[[FloatArgType], FloatArgType],
        fun_params0: pn.RandomVariable,
        current_iter: IntArgType,
    ) -> Tuple[bool, Union[str, None]]:
        """
        Termination based on numerical uncertainty about the parameters.

        Parameters
        ----------
        fun :
            One-dimensional objective function.
        fun_params0 :
            Belief over the parameters of the quadratic objective.
        current_iter :
            Current iteration of the PN method.
        """
        # Uncertainty over parameters given by the trace of the covariance.
        trace_cov = np.trace(fun_params0.cov)
        if trace_cov < self.abstol:
            return True, "uncertainty_abstol"
        elif trace_cov < np.linalg.norm(fun_params0.mean, ord=2) ** 2 * self.reltol:
            return True, "uncertainty_reltol"
        else:
            return False, None


class MaximumIterations(QuadOptStoppingCriterion):
    """
    Stopping criterion based on a maximum number of iterations.

    Parameters
    ----------
    maxiter :
        Maximum number of iterations
    """

    def __init__(self, maxiter: IntArgType):
        self.maxiter = maxiter
        super().__init__(stopping_criterion=self._maxiter_stopping_criterion)

    def _maxiter_stopping_criterion(
        self,
        fun: Callable[[FloatArgType], FloatArgType],
        fun_params0: pn.RandomVariable,
        current_iter: IntArgType,
    ) -> Tuple[bool, Union[str, None]]:
        """
        Termination based on maximum number of iterations.

        Parameters
        ----------
        fun :
            One-dimensional objective function.
        fun_params0 :
            Belief over the parameters of the quadratic objective.
        current_iter :
            Current iteration of the PN method.
        """
        if current_iter >= self.maxiter:
            return True, "maxiter"
        else:
            return False, None


class Residual(QuadOptStoppingCriterion):
    """
    StoppingCriterion based on the residual.

    Stop iterating whenever :math:`\\lVert f(x_*) \\rVert \\leq \\min(\\text{abstol}`.

    Parameters
    ----------
    abstol :
        Absolut residual tolerance.
    reltol :
        Relative residual tolerance.
    """

    def __init__(self, abstol: FloatArgType, reltol: FloatArgType):
        self.abstol = abstol
        self.reltol = reltol
        super().__init__(stopping_criterion=self._residual_stopping_criterion)

    def _residual_stopping_criterion(
        self,
        fun: Callable[[FloatArgType], FloatArgType],
        fun_params0: pn.RandomVariable,
        current_iter: IntArgType,
    ) -> Tuple[bool, Union[str, None]]:
        """
        Termination based on the residual.

        Parameters
        ----------
        fun :
            One-dimensional objective function.
        fun_params0 :
            Belief over the parameters of the quadratic objective.
        current_iter :
            Current iteration of the PN method.
        """
        a, b, _ = fun_params0.mean
        x_opt_estimate = -b / a
        residual = fun(x_opt_estimate)
        if residual < self.abstol:
            return True, "residual_abstol"
        elif residual < fun(1.0) ** 2 * self.reltol:
            return True, "residual_reltol"
        else:
            return False, None
