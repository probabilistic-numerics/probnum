"""Utility functions for arrays and the like."""

import numpy as np
import scipy.sparse
import probnum

__all__ = [
    "atleast_1d",
    "atleast_2d",
    "as_colvec",
    "assert_is_1d_ndarray",
    "assert_is_2d_ndarray",
]


def atleast_1d(*rvs):
    """
    Convert arrays or random variables to arrays or random variables
    with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved. Sparse arrays are not
    transformed.

    Parameters
    ----------
    rvs: array-like or RandomVariable
        One or more input random variables or arrays.

    Returns
    -------
    res : array-like or list
        An array / random variable or list of arrays / random variables,
        each with ``a.ndim >= 1``.

    See Also
    --------
    atleast_2d

    """
    res = []
    for rv in rvs:
        if isinstance(rv, scipy.sparse.spmatrix):
            result = rv
        elif isinstance(rv, np.ndarray):
            result = np.atleast_1d(rv)
        elif isinstance(rv, probnum.RandomVariable):
            raise NotImplementedError
        else:
            result = rv
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def atleast_2d(*rvs):
    """
    View inputs as arrays with at least two dimensions.

    Parameters
    ----------
    arys1, arys2, ... : array-like or RandomVariable
        One or more random variables or array-like sequences. Non-array
        inputs are converted to arrays or random variables. Arrays or
        random variables that already have two or more dimensions are
        preserved.

    Returns
    -------
    rvs: array-like or list
        An array, random variable or a list of arrays / random
        variables, each with ``a.ndim >= 2``.

    See Also
    --------
    atleast_1d
    """
    res = []
    for rv in rvs:
        if isinstance(rv, scipy.sparse.spmatrix):
            result = rv
        elif isinstance(rv, np.ndarray):
            result = np.atleast_2d(rv)
        elif isinstance(rv, probnum.RandomVariable):
            raise NotImplementedError
        else:
            result = rv
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def as_colvec(vec):
    """
    Transform the given vector or random variable to column format.

    Given a vector (or random variable) of dimension (n,) return an array with
    dimensions (n, 1) instead. Higher-dimensional arrays are not changed.

    Parameters
    ----------
    vec : np.ndarray or RandomVariable
        Vector, array or random variable to be viewed as a column vector.

    Returns
    -------
    vec2d : np.ndarray or RandomVariable
    """
    if isinstance(vec, probnum.RandomVariable):
        if vec.shape != (vec.shape[0], 1):
            vec.reshape(newshape=(vec.shape[0], 1))
    else:
        if vec.ndim == 1:
            return vec[:, None]
    return vec


def assert_is_1d_ndarray(arr):
    """
    Checks whether arr is an (d,) np.ndarray.

    Used in classic optimization and mcmc, for instance.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Please enter arr of shape (d,)")
    elif len(arr.shape) != 1:
        raise ValueError("Please enter arr of shape (d,)")


def assert_is_2d_ndarray(arr):
    """
    Checks whether ar is an (n, d) np.ndarray.

    Used in classic optimization and mcmc, for instance.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Please enter arr of shape (n, d)")
    elif arr.ndim != 2:
        raise ValueError("Please enter arr of shape (n, d)")
