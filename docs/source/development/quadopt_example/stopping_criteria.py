"""Stopping criterion for a 1D quadratic optimization problem."""

from typing import Callable, Tuple, Union

import numpy as np

from probnum import randvars
from probnum.typing import FloatArgType, IntArgType


def parameter_uncertainty(
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: randvars.RandomVariable,
    current_iter: IntArgType,
    abstol: FloatArgType,
    reltol: FloatArgType,
) -> Tuple[bool, Union[str, None]]:
    """Termination based on numerical uncertainty about the parameters.

    Parameters
    ----------
    fun :
        One-dimensional objective function.
    fun_params0 :
        Belief over the parameters of the quadratic objective.
    current_iter :
        Current iteration of the PN method.
    abstol :
        Absolute convergence tolerance.
    reltol :
        Relative convergence tolerance.
    """
    # Uncertainty over parameters given by the trace of the covariance.
    trace_cov = np.trace(fun_params0.cov)
    if trace_cov < abstol:
        return True, "uncertainty_abstol"
    elif trace_cov < np.linalg.norm(fun_params0.mean, ord=2) ** 2 * reltol:
        return True, "uncertainty_reltol"
    else:
        return False, None


def maximum_iterations(
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: randvars.RandomVariable,
    current_iter: IntArgType,
    maxiter: IntArgType,
) -> Tuple[bool, Union[str, None]]:
    """Termination based on maximum number of iterations.

    Parameters
    ----------
    fun :
        One-dimensional objective function.
    fun_params0 :
        Belief over the parameters of the quadratic objective.
    current_iter :
        Current iteration of the PN method.
    maxiter :
        Maximum number of iterations.
    """
    if current_iter >= maxiter:
        return True, "maxiter"
    else:
        return False, None


def residual(
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: randvars.RandomVariable,
    current_iter: IntArgType,
    abstol: FloatArgType,
    reltol: FloatArgType,
) -> Tuple[bool, Union[str, None]]:
    """Termination based on the residual.

    Stop iterating whenever :math:`\\lVert f(x_*) \\rVert \\leq \\min(\\text{abstol}`.

    Parameters
    ----------
    fun :
        One-dimensional objective function.
    fun_params0 :
        Belief over the parameters of the quadratic objective.
    current_iter :
        Current iteration of the PN method.
    abstol :
        Absolute convergence tolerance.
    reltol :
        Relative convergence tolerance.
    """
    a, b, _ = fun_params0.mean
    x_opt_estimate = -b / a
    resid = fun(x_opt_estimate)
    if resid < abstol:
        return True, "residual_abstol"
    elif resid < fun(1.0) ** 2 * reltol:
        return True, "residual_reltol"
    else:
        return False, None
