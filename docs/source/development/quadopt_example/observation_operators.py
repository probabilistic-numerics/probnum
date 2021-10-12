"""Observation operators for 1D quadratic optimization."""

from typing import Callable

import numpy as np

from probnum import utils
from probnum.typing import FloatArgType


def function_evaluation(
    fun: Callable[[FloatArgType], FloatArgType], action: FloatArgType
) -> np.float_:
    """Observe a (noisy) function evaluation of the quadratic objective.

    Parameters
    ----------
    fun :
        Quadratic objective function to optimize.
    action :
        Input to the objective function.
    """
    observation = fun(action)
    try:
        return utils.as_numpy_scalar(observation, dtype=np.floating)
    except TypeError as exc:
        raise TypeError(
            "The given argument `p` can not be cast to a `np.floating` object."
        ) from exc
