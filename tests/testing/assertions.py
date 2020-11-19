"""This module implements custom assertion classes for test cases in unittest."""

import numpy as np

__all__ = ["NumpyAssertions"]


class NumpyAssertions:
    """Wraps numpy's assert statements for comparing arrays."""

    __unittest = True  # avoids printing traceback from this class

    def assertApproxEqual(self, actual, desired, significant=7, msg=""):
        """Raises an AssertionError if two items are not equal up to significant digits.

        .. note:: It is recommended to use one of `assert_allclose`,
                  `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
                  instead of this function for more consistent floating point
                  comparisons.

        Given two numbers, check that they are approximately equal.
        Approximately equal is defined as the number of significant digits
        that agree.

        Parameters
        ----------
        actual : scalar
            The object to check.
        desired : scalar
            The expected object.
        significant : int, optional
            Desired precision, default is 7.
        msg : str, optional
            The error message to be printed in case of failure.

        Raises
        ------
        AssertionError
          If actual and desired are not equal up to specified precision.
        """
        np.testing.assert_approx_equal(
            actual=actual,
            desired=desired,
            significant=significant,
            err_msg=msg,
            verbose=True,
        )

    def assertAllClose(
        self, actual, desired, rtol=1e-7, atol=0, equal_nan=True, msg=""
    ):
        """Raises an AssertionError if two objects are not equal up to desired
        tolerance.

        The test is equivalent to ``allclose(actual, desired, rtol, atol)`` (note
        that ``allclose`` has different default values). It compares the difference
        between `actual` and `desired` to ``atol + rtol * abs(desired)``.

        Parameters
        ----------
        actual : array_like
            Array obtained.
        desired : array_like
            Array desired.
        rtol : float, optional
            Relative tolerance.
        atol : float, optional
            Absolute tolerance.
        equal_nan : bool, optional.
            If True, NaNs will compare equal.
        msg : str, optional
            The error message to be printed in case of failure.

        Raises
        ------
        AssertionError
            If actual and desired are not equal up to specified precision.
        """
        np.testing.assert_allclose(
            actual=actual,
            desired=desired,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            err_msg=msg,
        )

    def assertArrayAlmostEqualNulp(self, actual, desired, nulp=1):
        """Compare two arrays relatively to their spacing.

        This is a relatively robust method to compare two arrays whose amplitude
        is variable.

        Parameters
        ----------
        actual : array_like
            Array obtained.
        desired : array_like
            Array desired.
        nulp : int, optional
            The maximum number of unit in the last place for tolerance (see Notes).
            Default is 1.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the spacing between ``actual`` and ``desired`` for one or more elements
            is larger than ``nulp``.

        Notes
        -----
        An assertion is raised if the following condition is not met::

            abs(x - y) <= nulps * spacing(maximum(abs(x), abs(y)))
        """
        np.testing.assert_array_almost_equal_nulp(x=actual, y=desired, nulp=nulp)

    def assertArrayMaxUlp(self, actual, desired, maxulp=1, dtype=None):
        """Check that all items of arrays differ in at most N Units in the Last Place.

        Parameters
        ----------
        actual : array_like
            Array obtained.
        desired : array_like
            Array desired.
        maxulp : int, optional
            The maximum number of units in the last place that elements of `a` and
            `b` can differ. Default is 1.
        dtype : dtype, optional
            Data-type to convert `a` and `b` to if given. Default is None.

        Returns
        -------
        ret : ndarray
            Array containing number of representable floating point numbers between
            items in ``actual`` and ``desired``.

        Raises
        ------
        AssertionError
            If one or more elements differ by more than ``maxulp``.
        """
        np.testing.assert_array_max_ulp(a=actual, b=desired, maxulp=maxulp, dtype=dtype)

    def assertArrayEqual(self, actual, desired, msg=""):
        """Raises an AssertionError if two array_like objects are not equal.

        Given two array_like objects, check that the shape is equal and all
        elements of these objects are equal. An exception is raised at
        shape mismatch or conflicting values. In contrast to the standard usage
        in numpy, NaNs are compared like numbers, no assertion is raised if
        both objects have NaNs in the same positions.
        The usual caution for verifying equality with floating point numbers is
        advised.

        Parameters
        ----------
        actual : array_like
            The actual object to check.
        desired : array_like
            The desired, expected object.
        msg : str, optional
            The error message to be printed in case of failure.

        Raises
        ------
        AssertionError
            If actual and desired objects are not equal.
        """
        np.testing.assert_array_equal(x=actual, y=desired, err_msg=msg, verbose=True)

    def assertArrayLess(self, smaller, larger, msg=""):
        """Raises an AssertionError if two array_like objects are not ordered by less
        than.

        Given two array_like objects, check that the shape is equal and all
        elements of the first object are strictly smaller than those of the
        second object. An exception is raised at shape mismatch or incorrectly
        ordered values. Shape mismatch does not raise if an object has zero
        dimension. In contrast to the standard usage in numpy, NaNs are
        compared, no assertion is raised if both objects have NaNs in the same
        positions.

        Parameters
        ----------
        smaller : array_like
          The smaller object to check.
        larger : array_like
          The larger object to compare.
        msg : string
          The error message to be printed in case of failure.

        Raises
        ------
        AssertionError
            If actual and desired objects are not equal.
        """
        np.testing.assert_array_less(x=smaller, y=larger, err_msg=msg, verbose=True)
