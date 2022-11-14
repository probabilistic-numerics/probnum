from typing import Union

import numpy as np

from probnum import backend, linops

from . import _core

__all__ = [
    "assert_allclose",
    "assert_array_equal",
    "assert_equal",
]


def assert_equal(
    actual: Union[backend.Array, linops.LinearOperator],
    desired: Union[backend.Array, linops.LinearOperator],
    /,
    *,
    err_msg: str = "",
    verbose: bool = True,
):
    """Raises an AssertionError if two objects are not equal.

    Given two objects (scalars, lists, tuples, dictionaries,
    :class:`~probnum.backend.Array`\s, :class:`~probnum.linops.LinearOperator`\s),
    check that all elements of these objects are equal. An exception is raised
    at the first conflicting values.

    When one of ``actual`` and ``desired`` is a scalar and the other is array_like,
    the function checks that each element of the array_like object is equal to
    the scalar.

    This function handles NaN comparisons as if NaN was a "normal" number.
    That is, AssertionError is not raised if both objects have NaNs in the same
    positions.  This is in contrast to the IEEE standard on NaNs, which says
    that NaN compared to anything must return False.

    Parameters
    ----------
    actual
        The object to check.
    desired
        The expected object.
    err_msg
        The error message to be printed in case of failure.
    verbose
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal.
    """
    np.testing.assert_equal(
        *_core.to_numpy(actual, desired), err_msg=err_msg, verbose=verbose
    )


def assert_allclose(
    actual: Union[backend.Array, linops.LinearOperator],
    desired: Union[backend.Array, linops.LinearOperator],
    /,
    *,
    rtol: float = 1e-7,
    atol: float = 0,
    equal_nan: bool = True,
    err_msg: str = "",
    verbose: bool = True,
):
    """Raises an AssertionError if two objects are not equal up to desired tolerance.

    The test compares the difference
    between `actual` and `desired` to ``atol + rtol * abs(desired)``.

    Parameters
    ----------
    actual
        Array obtained.
    desired
        Array desired.
    rtol
        Relative tolerance.
    atol
        Absolute tolerance.
    equal_nan
        If True, NaNs will compare equal.
    err_msg
        The error message to be printed in case of failure.
    verbose
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.
    """
    np.testing.assert_allclose(
        *_core.to_numpy(actual, desired),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        err_msg=err_msg,
        verbose=verbose,
    )


def assert_array_equal(
    actual: Union[backend.Array, linops.LinearOperator],
    desired: Union[backend.Array, linops.LinearOperator],
    /,
    *,
    err_msg: str = "",
    verbose: bool = True,
):
    """Raises an AssertionError if two array_like objects are not equal.

    Given two array_like objects, check that the shape is equal and all
    elements of these objects are equal (but see the Notes for the special
    handling of a scalar). An exception is raised at shape mismatch or
    conflicting values. In contrast to the standard usage in numpy, NaNs
    are compared like numbers, no assertion is raised if both objects have
    NaNs in the same positions.

    Parameters
    ----------
    actual
        The actual object to check.
    desired
        The desired, expected object.
    err_msg
        The error message to be printed in case of failure.
    verbose
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired objects are not equal.
    """
    np.testing.assert_array_equal(
        *_core.to_numpy(actual, desired), err_msg=err_msg, verbose=verbose
    )
