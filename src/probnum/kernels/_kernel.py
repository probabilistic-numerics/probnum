"""Kernel or covariance function."""

from typing import Callable, Generic, Optional, TypeVar

import numpy as np
import scipy.spatial

import probnum.utils as _utils
from probnum.type import IntArgType

_InputType = TypeVar("InputType")


class Kernel(Generic[_InputType]):
    """Kernel or covariance function.

    Kernels describes the spatial or temporal variation of a random process. If
    evaluated at two sets of points a kernel is defined as the covariance of the
    values of the random process at these locations.

    Parameters
    ----------
    fun :
        Function :math:`k(x_0,x_1)` defining the kernel. Assumed to be non-vectorized.
    output_dim :
        Output dimension of the kernel. If larger than 1, ``fun`` must return a
        matrix of dimension *(output_dim, output_dim)* representing the covariance at
        the inputs.
    """

    def __init__(
        self,
        fun: Callable[[_InputType, _InputType], np.float_],
        output_dim: IntArgType = 1,
    ):
        self.__fun = fun
        self.__output_dim = output_dim

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    # todo: replace fun argument with clsmethod from_function?
    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        """Evaluate the kernel.

        If only the first input is provided the kernel matrix :math:`K=k(X_0, X_0)` is
        computed.

        Parameters
        ----------
        x0 :
            First input.
        x1 :
            Second input.

        Returns
        -------
        cov :
            *shape=(n0, n1) or (n0, n1, output_dim, output_dim)* -- Kernel evaluated
            pairwise for all entries / rows of ``x0`` and ``x1``.
        """
        if x1 is None:
            x1 = x0

        x0 = _utils.as_colvec(x0)
        x1 = _utils.as_colvec(x1)

        return scipy.spatial.distance.cdist(x0, x1, metric=self.__fun)

    @property
    def output_dim(self) -> int:
        """Dimension of the evaluated covariance function.

        The resulting kernel matrix :math:`k(x_0, x_1) \\in
        \\mathbb{R}^{n_out \\times n_out}` has dimension *(output_dim,
        output_dim)*.
        """
        return self.__output_dim
