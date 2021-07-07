"""Policies for 1D quadratic optimization."""

from typing import Callable, Optional

import numpy as np

from probnum import randvars
from probnum.typing import FloatArgType


def explore_exploit_policy(
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: randvars.RandomVariable,
    rng: np.random.Generator,
) -> float:
    """Policy exploring around the estimate of the minimum based on the certainty about
    the parameters.

    Parameters
    ----------
    fun :
        One-dimensional objective function.
    fun_params0 :
        Belief over the parameters of the quadratic objective.
    rng :
        Random number generator.
    """
    a0, b0, _ = fun_params0
    sample = randvars.Normal(0, np.trace(fun_params0.cov)).sample(rng=rng)
    return -b0.mean / a0.mean + sample


def stochastic_policy(
    fun: Callable[[FloatArgType], FloatArgType],
    fun_params0: randvars.RandomVariable,
    rng: np.random.Generator,
) -> float:
    """Policy returning a random action.

    Parameters
    ----------
    fun :
        One-dimensional objective function.
    fun_params0 :
        Belief over the parameters of the quadratic objective.
    rng :
        Random number generator.
    """
    return randvars.Normal(mean=0.0, cov=1.0).sample(rng=rng)
