"""
Utility functions for functions, methods and the like.
"""
from typing import Any, Callable

import numpy as np


def assert_evaluates_to_scalar(fun: Callable[[Any], Any], valid_input: Any) -> None:
    """
    Checks whether the output of a function is a scalar.

    Parameters
    ----------
    fun :
        Function.
    valid_input :
        Function input.

    Raises
    ------
    ValueError
        If the function does not evaluate to a scalar.
    """
    if not np.isscalar(fun(valid_input)):
        raise ValueError("Function does not evaluate to scalar.")
