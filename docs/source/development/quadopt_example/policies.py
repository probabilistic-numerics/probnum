"""Policies for 1D quadratic optimization."""

from typing import Callable, Optional

import numpy as np

from probnum import randvars
from probnum.type import FloatArgType, RandomStateArgType


def explore_exploit_policy(
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: randvars.RandomVariable,
    random_state: RandomStateArgType = None,
) -> float:
    """Policy exploring around the estimate of the minimum based on the certainty about
    the parameters.

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
    a0, b0, _ = fun_params0
    return (
        -b0.mean / a0.mean
        + randvars.Normal(
            0, np.trace(fun_params0.cov), random_state=random_state
        ).sample()
    )


def stochastic_policy(
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: randvars.RandomVariable,
    random_state: RandomStateArgType = None,
) -> float:
    """Policy returning a random action.

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
    return randvars.Normal(mean=0.0, cov=1.0, random_state=random_state).sample()
