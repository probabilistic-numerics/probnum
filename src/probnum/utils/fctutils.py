"""
Utility functions for functions, methods and the like.
"""

import numpy as np


__all__ = ["assert_evaluates_to_scalar"]


def assert_evaluates_to_scalar(fct, valid_input):
    """
    Checks whether the output of a function is a scalar.
    """
    if not np.isscalar(fct(valid_input)):
        raise ValueError("Function does not evaluate to scalar.")
