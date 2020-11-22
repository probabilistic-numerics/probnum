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
        Function :math:`k: \\mathbb{R}^{d_{in}} \\times \\mathbb{R}^{d_{in}} \\rightarrow
        \\mathbb{R}^{d_{out} \\times d_{out}}` defining the kernel. Assumed to be
        vectorized and callable with the second argument being `None`.
    output_dim :
        Output dimension of the kernel. If larger than 1, ``fun`` must return a
        matrix of dimension *(output_dim, output_dim)* representing the covariance at
        the inputs.

    See Also
    --------
    askernel : Convert a callable to a :class:`Kernel`.

    Examples
    --------
    >>> import numpy as np
    >>> import probnum as pn
    >>> from probnum.kernels import Kernel
    >>> # Data
    >>> x = np.array([[1, 2], [-1, -1]])
    >>> # Custom kernel from a (non-vectorized) covariance function
    >>> k = pn.askernel(lambda x0, x1: (x0.T @ x1 - 1.0) ** 2)
    >>> k(x)
    array([[16., 16.],
           [16.,  1.]])
    >>> # Custom kernel implemented more efficiently via vectorization
    >>> def custom_kernel_fun(x0, x1=None):
    ...     if x1 is None:
    ...         x1 = x0
    ...     return (x0 @ x1.T - 1.0) ** 2
    >>> k = Kernel(output_dim=1, kernel=custom_kernel_fun)
    >>> k(x)
    array([[16., 16.],
           [16.,  1.]])
    """

    def __init__(
        self,
        output_dim: IntArgType,
        kernel: Optional[
            Callable[[_InputType, Optional[_InputType]], np.ndarray]
        ] = None,
    ):
        self.__kernel = kernel
        self._output_dim = output_dim

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        """Evaluate the kernel.

        Computes the covariance function at ``x0`` and ``x1``. If the inputs have
        more than one dimension the covariance function is evaluated pairwise for all
        observations determined by the first dimension of ``x0`` and ``x1``. If
        only ``x0`` is given the kernel matrix :math:`K=k(X_0, X_0)` is computed.

        Parameters
        ----------
        x0 :
            *shape=(input_dim,) or (n0, input_dim)* -- First input.
        x1 :
            *shape=(input_dim,) or (n1, input_dim)* -- Second input.

        Returns
        -------
        cov :
            *shape=(output_dim, output_dim) or (n0, n1) or (n0, n1, output_dim,
            output_dim)* -- Kernel evaluated at ``x0`` and ``x1`` or kernel matrix
            containing pairwise evaluations for all observations in ``x0`` and ``x1``.
        """
        if self.__kernel is not None:
            return self.__kernel(x0, x1)
        else:
            raise NotImplementedError

    @property
    def output_dim(self) -> int:
        """Dimension of the evaluated covariance function.

        The resulting evaluated kernel :math:`k(x_0, x_1) \\in
        \\mathbb{R}^{d_{out} \\times d_{out}}` has *shape=(output_dim,
        output_dim)*.
        """
        return self._output_dim
