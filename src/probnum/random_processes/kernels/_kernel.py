"""Kernel or covariance function."""

from typing import Callable, Generic, Optional, TypeVar

import numpy as np
import scipy.spatial

_InputType = TypeVar("InputType")


class Kernel(Generic[_InputType]):
    """Kernel or covariance function.

    Kernels describes the spatial or temporal variation of a random process. If
    evaluated at two sets of points a kernel is defined as the covariance of the
    values of the random process at these locations.

    Parameters
    ----------
    fun :
        Function :math:`k(x,y)` defining the kernel.
    """

    def __init__(self, fun: Callable[[_InputType, _InputType], np.float_]):
        self._fun = fun

    def __call__(self, x0: [_InputType], x1: Optional[_InputType] = None) -> np.ndarray:
        """Compute the kernel matrix.

        If only the first input is provided the kernel matrix :math:`K(X_0, X_0)` is
        computed.

        Parameters
        ----------
        x0 :
            First input.
        x1 :
            Second input.
        """
        if x1 is None:
            x1 = x0
        return scipy.spatial.distance.cdist(x0, x1, metric=self._fun)
