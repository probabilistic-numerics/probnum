"""Kernel / covariance function."""

import abc
from typing import Generic, Optional, Tuple, TypeVar, Union

import numpy as np

import probnum.utils as _utils
from probnum.typing import IntArgType, ShapeArgType, ShapeType

_InputType = TypeVar("InputType")


class Kernel(Generic[_InputType], abc.ABC):
    """Kernel / covariance function.

    Abstract base class for kernels / covariance functions. Kernels are a
    generalization of a positive-definite function or matrix. They
    typically define the covariance function of a random process and thus describe
    its spatial or temporal variation. If evaluated at two sets of points a kernel
    gives the covariance of the random process at these locations.

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
    output_dim :
        Output dimension of the kernel.

    Examples
    --------
    Kernels are implemented by subclassing this abstract base class.

    >>> from probnum.kernels import Kernel
    ...
    >>> class CustomLinearKernel(Kernel):
    ...
    ...     def __init__(self, constant=0.0):
    ...         self.constant = constant
    ...         super().__init__(input_dim=1, output_dim=1)
    ...
    ...     def __call__(self, x0, x1=None):
    ...         # Check and reshape inputs
    ...         x0, x1, kernshape = self._check_and_reshape_inputs(x0, x1)
    ...
    ...         # Compute kernel matrix
    ...         if x1 is None:
    ...             x1 = x0
    ...         kernmat = x0 @ x1.T + self.constant
    ...
    ...         return Kernel._reshape_kernelmatrix(kernmat, newshape=kernshape)

    We can now evaluate the kernel like so.

    >>> import numpy as np
    >>> k = CustomLinearKernel(constant=1.0)
    >>> k(np.linspace(0, 1, 4)[:, None])
    array([[1.        , 1.        , 1.        , 1.        ],
           [1.        , 1.11111111, 1.22222222, 1.33333333],
           [1.        , 1.22222222, 1.44444444, 1.66666667],
           [1.        , 1.33333333, 1.66666667, 2.        ]])
    """

    # pylint: disable="invalid-name"
    def __init__(
        self,
        input_dim: IntArgType,
        output_dim: IntArgType = 1,
    ):
        self._input_dim = np.int_(_utils.as_numpy_scalar(input_dim))
        self._output_dim = np.int_(_utils.as_numpy_scalar(output_dim))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    @abc.abstractmethod
    def __call__(
        self, x0: _InputType, x1: Optional[_InputType] = None
    ) -> Union[np.ndarray, np.float_]:
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
            *shape=(), (output_dim, output_dim) or (n0, n1) or (n0, n1, output_dim,
            output_dim)* -- Kernel evaluated at ``x0`` and ``x1`` or kernel matrix
            containing pairwise evaluations for all observations in ``x0`` (and ``x1``).
        """
        raise NotImplementedError

    @property
    def input_dim(self) -> int:
        """Dimension of arguments of the covariance function.

        The dimension of inputs to the covariance function :math:`k : \\mathbb{R}^{
        d_{in}} \\times \\mathbb{R}^{d_{in}} \\rightarrow
        \\mathbb{R}^{d_{out} \\times d_{out}}`.
        """
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Dimension of the evaluated covariance function.

        The resulting evaluated kernel :math:`k(x_0, x_1) \\in
        \\mathbb{R}^{d_{out} \\times d_{out}}` has *shape=(output_dim,
        output_dim)*.
        """
        return self._output_dim

    def _check_and_reshape_inputs(
        self,
        x0: _InputType,
        x1: Optional[_InputType] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], ShapeType]:
        """Check and transform inputs of the covariance function.

        Checks the shape of the inputs to the covariance function and
        transforms the inputs into two-dimensional :class:`numpy.ndarray`s such that
        inputs are stacked row-wise.

        Parameters
        ----------
        x0 :
            First input to the covariance function.
        x1 :
            Second input to the covariance function.

        Returns
        -------
        x0 :
            First input to the covariance function.
        x1 :
            Second input to the covariance function.
        kernshape :
            Shape of the evaluation of the covariance function.

        Raises
        -------
        ValueError :
            If input shapes of x0 and x1 do not match the kernel input dimension or
            each other.
        """
        # pylint: disable="too-many-boolean-expressions"

        # Check and promote shapes
        x0 = np.asarray(x0)
        if x1 is None:
            if (
                (x0.ndim == 0 and self.input_dim > 1)  # Scalar input
                or (x0.ndim == 1 and x0.shape[0] != self.input_dim)  # Vector input
                or (x0.ndim >= 2 and x0.shape[1] != self.input_dim)  # Matrix input
            ):
                raise ValueError(
                    f"Argument shape x0.shape={x0.shape} does not match "
                    "kernel input dimension."
                )

            # Determine correct shape for the kernel matrix as the output of __call__
            kernshape = self._get_shape_kernelmatrix(
                x0_shape=x0.shape, x1_shape=x0.shape
            )

            return np.atleast_2d(x0), None, kernshape
        else:
            x1 = np.asarray(x1)
            err_msg = (
                f"Argument shapes x0.shape={x0.shape} and x1.shape="
                f"{x1.shape} do not match kernel input dimension "
                f"{self.input_dim}. Try passing either two vectors or two "
                "matrices with the second dimension equal to the kernel input "
                "dimension."
            )

            # pylint: disable=redefined-variable-type

            # Promote unequal shapes
            if x0.ndim < 2 and x1.ndim == 2:
                x0 = np.atleast_2d(x0)
            if x1.ndim < 2 and x0.ndim == 2:
                x1 = np.atleast_2d(x1)
            if x0.ndim != x1.ndim:  # Shape mismatch
                raise ValueError(err_msg)

            # Check shapes
            if (
                (x0.ndim == 0 and self.input_dim > 1)  # Scalar input
                or (
                    x0.ndim == 1  # Vector input
                    and not (x0.shape[0] == x1.shape[0] == self.input_dim)
                )
                or (
                    x0.ndim == 2  # Matrix input
                    and not (x0.shape[1] == x1.shape[1] == self.input_dim)
                )
            ):
                raise ValueError(err_msg)

            # Determine correct shape for the kernel matrix as the output of __call__
            kernshape = self._get_shape_kernelmatrix(
                x0_shape=x0.shape, x1_shape=x1.shape
            )

            return np.atleast_2d(x0), np.atleast_2d(x1), kernshape

    def _get_shape_kernelmatrix(
        self,
        x0_shape: ShapeArgType,
        x1_shape: ShapeArgType,
    ) -> ShapeType:
        """Determine the shape of the kernel matrix based on the given arguments.

        Determine the correct shape of the covariance function evaluated at the given
        input arguments. If inputs are vectors the output is a numpy scalar if the
        output dimension of the kernel is 1, otherwise *shape=(output_dim,
        output_dim)*. If inputs represent multiple observations, then the resulting
        matrix has *shape=(n0, n1) or (n0, n1, output_dim, output_dim)*.

        Parameters
        ----------
        x0_shape :
            Shape of the first input to the covariance function.
        x1_shape :
            Shape of the second input to the covariance function.
        """
        if len(x0_shape) <= 1 and len(x1_shape) <= 1:
            if self.output_dim == 1:
                kern_shape = 0
            else:
                kern_shape = ()
        else:
            kern_shape = (x0_shape[0], x1_shape[0])

        if self.output_dim > 1:
            kern_shape += (
                self.output_dim,
                self.output_dim,
            )

        return _utils.as_shape(kern_shape)

    @staticmethod
    def _reshape_kernelmatrix(
        kerneval: np.ndarray, newshape: ShapeArgType
    ) -> np.ndarray:
        """Reshape the evaluation of the covariance function.

        Reshape the given evaluation of the covariance function to the correct shape,
        determined by the inputs x0 and x1. This method is designed to be called by
        subclasses of :class:`Kernel` in their :meth:`__call__` function to ensure
        the returned quantity has the correct shape independent of the implementation of
        the kernel.

        Parameters:
        -----------
        kerneval
            Covariance function evaluated at ``x0`` and ``x1``.
        newshape :
            New shape of the evaluation of the covariance function.
        """
        if newshape[0] == 0:
            return _utils.as_numpy_scalar(kerneval.squeeze())
        else:
            return kerneval.reshape(newshape)
