"""Kernel / covariance function."""

from typing import Callable, Generic, Optional, TypeVar

import numpy as np

from probnum.type import IntArgType

_InputType = TypeVar("InputType")


class Kernel(Generic[_InputType]):
    """Kernel / covariance function.

    Kernels describes the spatial or temporal variation of a random process. If
    evaluated at two sets of points a kernel is defined as the covariance of the
    values of the random process at these locations.

    Parameters
    ----------
    kernel :
        Function :math:`k: \\mathbb{R}^n \\times \\mathbb{R}^n \\rightarrow \\mathbb{
        R}^{n_{out} \\times n_{out}}` defining the kernel. Assumed to be vectorized
        and callable with the second argument being `None`.
    output_dim :
        Output dimension of the kernel. If larger than 1, ``fun`` must return a
        matrix of dimension *(output_dim, output_dim)* representing the covariance at
        the inputs.

    See Also
    --------
    askernel : Convert a bivariate function to a :class:`Kernel`.
    """

    def __init__(
        self,
        kernel: Callable[[_InputType, Optional[_InputType]], np.ndarray],
        output_dim: IntArgType = 1,
    ):
        self.__kernel = kernel
        self._output_dim = output_dim

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        """Evaluate the kernel.

        Computes the covariance function at ``x0`` and ``x1``. If the inputs have
        more than one dimension the covariance function is evaluated pairwise for all
        observations determined by the first dimension(s) of ``x0`` and ``x1``. If
        only ``x0`` is given the kernel matrix :math:`K=k(X_0, X_0)` is computed.

        Parameters
        ----------
        x0 :
            *shape=(n0,) or (n0, d)* -- First input.
        x1 :
            *shape=(n1,) or (n1, d)* -- Second input.

        Returns
        -------
        cov :
            *shape=(n0, n1) or (n0, n1, output_dim, output_dim)* -- Kernel evaluated
            at ``x0`` and ``x1`` or kernel matrix containing pairwise evaluations for
            all observations in ``x0`` and ``x1``.
        """
        return self.__kernel(x0, x1)

    @property
    def output_dim(self) -> int:
        """Dimension of the evaluated covariance function.

        The resulting kernel matrix :math:`k(x_0, x_1) \\in
        \\mathbb{R}^{n_{out} \\times n_{out}}` has dimension
        *(output_dim,output_dim)*.
        """
        return self._output_dim
