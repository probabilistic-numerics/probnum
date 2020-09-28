"""
Policies for 1D quadratic optimization.
"""

from typing import Callable, Optional

import numpy as np

import probnum as pn
import probnum.random_variables as rvs
from probnum.type import FloatArgType, RandomStateType


def stochastic_policy(
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: pn.RandomVariable,
    random_state: Optional[RandomStateType] = None,
) -> np.float_:
    """
    Policy exploring around the estimate of the minimum based on the certainty about the
    parameters.

    Parameters
    ----------
    fun :
        One-dimensional objective function.
    fun_params0 :
        Belief over the parameters of the quadratic objective.
    random_state :
        Random state of the policy. If None (or np.random), the global np.random state
        is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.
    """
    a0, b0, c0 = fun_params0
    return (
        -b0.mean / a0.mean
        + rvs.Normal(0, np.trace(fun_params0.cov), random_state=random_state).sample()
    )


def __nonzero_integer_gen(start=0):
    n = start
    while True:
        yield n
        n += 1


_nonzero_integer_gen = __nonzero_integer_gen()


def deterministic_policy(
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: pn.RandomVariable,
) -> float:
    """
    Policy returning the nonzero integers in sequence.

    Parameters
    ----------
    fun :
        One-dimensional objective function.
    fun_params0 :
        Belief over the parameters of the quadratic objective.
    """
    return next(_nonzero_integer_gen)
