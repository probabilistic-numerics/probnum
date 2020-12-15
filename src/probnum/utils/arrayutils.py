"""Utility functions for arrays and the like."""

from typing import Union

import numpy as np

import probnum.random_variables


def as_colvec(
    vec: Union[np.ndarray, "probnum.random_variables.RandomVariable"]
) -> Union[np.ndarray, "probnum.random_variables.RandomVariable"]:
    """Transform the given vector or random variable to column format.

    Given a vector (or random variable) of dimension (n,) return an array with
    dimensions (n, 1) instead. Higher-dimensional arrays are not changed.

    Parameters
    ----------
    vec
        Vector, array or random variable to be transformed into a column vector.
    """
    if isinstance(vec, probnum.random_variables.RandomVariable):
        if vec.shape != (vec.shape[0], 1):
            vec.reshape(newshape=(vec.shape[0], 1))
    else:
        if vec.ndim == 1:
            return vec[:, None]
    return vec
