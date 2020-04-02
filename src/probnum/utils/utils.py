"""Utility functions"""

import numpy as np
import scipy.sparse
import probnum.probability

__all__ = ["atleast_1d", "atleast_2d", "as_colvec"]


def atleast_1d(*rvs):
    """
    Convert arrays or random variables to arrays or random variables with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved. Sparse arrays
    are not transformed.

    Parameters
    ----------
    rvs: array-like or RandomVariable
        One or more input random variables or arrays.

    Returns
    -------
    res : array-like or list
        An array / random variable or list of arrays / random variables, each with ``a.ndim >= 1``.

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
        elif isinstance(rv, probnum.probability.RandomVariable):
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
        One or more random variables or array-like sequences. Non-array inputs are converted
        to arrays or random variables. Arrays or random variables that already have two or more dimensions are
        preserved.

    Returns
    -------
    rvs: array-like or list
        An array, random variable or a list of arrays / random variables, each with ``a.ndim >= 2``.

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
        elif isinstance(rv, probnum.probability.RandomVariable):
            raise NotImplementedError
        else:
            result = rv
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def as_colvec(arr):
    """
    Transform the given array to a row vector.

    Given a vector of dimension (n,) return an array with dimensions (n, 1) instead. Higher-dimensional arrays are not
    changed.

    Parameters
    ----------
    arr : np.ndarray
        Vector or array to be viewed as a column vector.

    Returns
    -------
    arr2d : np.ndarray
    """
    if arr.ndim == 1:
        return arr[:, None]
    else:
        return arr
