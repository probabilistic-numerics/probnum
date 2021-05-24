"""Utility functions for arrays and the like."""

from typing import Union

import numpy as np
import scipy

import probnum.randvars


def atleast_1d(*rvs):
    """Convert arrays or random variables to arrays or random variables with at least
    one dimension.

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
    """
    res = []
    for rv in rvs:
        if isinstance(rv, scipy.sparse.spmatrix):
            result = rv
        elif isinstance(rv, np.ndarray):
            result = np.atleast_1d(rv)
        elif isinstance(rv, probnum.randvars.RandomVariable):
            raise NotImplementedError
        else:
            result = rv
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def as_colvec(
    vec: Union[np.ndarray, "probnum.randvars.RandomVariable"]
) -> Union[np.ndarray, "probnum.randvars.RandomVariable"]:
    """Transform the given vector or random variable to column format. Given a vector
    (or random variable) of dimension (n,) return an array with dimensions (n, 1)
    instead. Higher-dimensional arrays are not changed.

    Parameters
    ----------
    vec
        Vector, array or random variable to be transformed into a column vector.
    """
    if isinstance(vec, probnum.randvars.RandomVariable):
        if vec.shape != (vec.shape[0], 1):
            vec.reshape(newshape=(vec.shape[0], 1))
    else:
        if vec.ndim == 1:
            return vec[:, None]
    return vec
