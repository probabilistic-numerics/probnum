"""Utility functions for arrays and the like."""

from typing import Union

import numpy as np

import probnum


def as_colvec(
    vec: Union[np.ndarray, "probnum.RandomVariable"]
) -> Union[np.ndarray, "probnum.RandomVariable"]:
    """
    Transform the given vector or random variable to column format.

    Given a vector (or random variable) of dimension (n,) return an array with
    dimensions (n, 1) instead. Higher-dimensional arrays are not changed.

    Parameters
    ----------
    vec
        Vector, array or random variable to be transformed into a column vector.
    """
    if isinstance(vec, probnum.RandomVariable):
        if vec.shape != (vec.shape[0], 1):
            vec.reshape(newshape=(vec.shape[0], 1))
    else:
        if vec.ndim == 1:
            return vec[:, None]
    return vec


def assert_is_1d_ndarray(arr: np.ndarray) -> None:
    """
    Checks whether ``arr`` is an (d,) np.ndarray.

    Parameters
    ----------
    arr
        Input array.

    Raises
    ------
    ValueError
        If the given object has a non-admissable shape.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Please enter arr of shape (d,)")
    elif len(arr.shape) != 1:
        raise ValueError("Please enter arr of shape (d,)")


def assert_is_2d_ndarray(arr: np.ndarray) -> None:
    """
    Checks whether ``arr`` is an (n, d) np.ndarray.

    Parameters
    ----------
    arr
        Input array.

    Raises
    ------
    ValueError
        If the given object has a non-admissable shape.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Please enter arr of shape (n, d)")
    elif arr.ndim != 2:
        raise ValueError("Please enter arr of shape (n, d)")
