"""Utility functions for kernels."""
from typing import Callable

import numpy as np
import scipy.spatial

from . import _kernel


def askernel(fun: Callable) -> _kernel.Kernel:
    """Convert ``fun`` to a :class:`Kernel`.

    Creates a kernel / covariance function from a (non-vectorized) function
    :math:`k : \\mathbb{R}^{d_{in}} \\times \\mathbb{R}^{d_{in}} \\rightarrow
    \\mathbb{R}`. When a kernel matrix is computed the given function ``fun`` is
    applied to all pairs of inputs.

    Parameters
    ----------
    fun :
        Callable to be represented as a :class:`Kernel`.

    See Also
    --------
    Kernel : Class representing kernels / covariance functions.

    Examples
    --------
    >>> import numpy as np
    >>> import probnum as pn
    >>> # Custom linear kernel
    >>> k = pn.askernel(lambda x0, x1: np.inner(x0, x1).squeeze())
    >>> # Data
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> # Kernel matrix
    >>> k(X)
    array([[14., 32.],
           [32., 77.]])
    """
    if callable(fun):

        def _kernfun_vectorized(x0, x1=None) -> np.ndarray:
            # pylint: disable=invalid-name
            x0 = np.atleast_2d(x0)
            if x1 is None:
                x1 = x0
            else:
                x1 = np.atleast_2d(x1)

            # Evaluate fun pairwise for all rows of x0 and x1
            return scipy.spatial.distance.cdist(x0, x1, metric=fun)

        return _kernel.Kernel(kernel=_kernfun_vectorized, output_dim=1)
    else:
        raise ValueError(
            f"Argument of type {type(fun)} is not callable and therefore cannot be "
            f"converted to a kernel."
        )
