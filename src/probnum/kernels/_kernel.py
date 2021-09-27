"""Kernel / covariance function."""

import abc
from typing import Generic, Optional, Tuple, TypeVar, Union

import numpy as np

import probnum.utils as _utils
from probnum.typing import IntArgType, ShapeType

_InputType = TypeVar("_InputType")


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
    ...     def _evaluate(self, x0, x1=None):
    ...         if x1 is None:
    ...             x1 = x0
    ...
    ...         return np.sum(x0 * x1, axis=-1)[..., None, None] + self.constant

    We can now evaluate the kernel like so.

    >>> import numpy as np
    >>> k = CustomLinearKernel(constant=1.0)
    >>> xs = np.linspace(0, 1, 4)[:, None]
    >>> k(xs[:, None, :], xs[None, :, :])
    array([[1.        , 1.        , 1.        , 1.        ],
           [1.        , 1.11111111, 1.22222222, 1.33333333],
           [1.        , 1.22222222, 1.44444444, 1.66666667],
           [1.        , 1.33333333, 1.66666667, 2.        ]])
    """

    def __init__(
        self,
        input_dim: IntArgType,
        output_dim: IntArgType = 1,
    ):
        self._input_dim = np.int_(_utils.as_numpy_scalar(input_dim))
        self._output_dim = np.int_(_utils.as_numpy_scalar(output_dim))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __call__(
        self,
        x0: _InputType,
        x1: Optional[_InputType] = None,
        squeeze_output_dim: bool = True,
    ) -> np.ndarray:
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

        x0, x1, broadcast_input_shape = self._check_and_reshape_inputs(x0, x1)

        cov = self._evaluate(x0, x1)

        assert cov.shape == broadcast_input_shape[:-1] + 2 * (self._output_dim,)

        if self.output_dim == 1 and squeeze_output_dim:
            cov = np.squeeze(cov, axis=(-2, -1))

        return cov

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

    @abc.abstractmethod
    def _evaluate(
        self,
        x0: _InputType,
        x1: Optional[_InputType] = None,
    ) -> Union[np.ndarray, np.float_]:
        pass

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

        err_msg = (
            "`{0}` does not broadcast to the kernel's input dimension "
            f"{self.input_dim} along the last axis, since "
            "`{0}.shape = {1}`."
        )

        x0 = np.atleast_1d(x0)

        if x0.shape[-1] not in (self.input_dim, 1):
            # This will never be called if the original input was a scalar
            raise ValueError(err_msg.format("x0", x0.shape))

        broadcast_input_shape = x0.shape

        if x1 is not None:
            x1 = np.atleast_1d(x1)

            if x1.shape[-1] not in (self.input_dim, 1):
                # This will never be called if the original input was a scalar
                raise ValueError(err_msg.format("x1", x1.shape))

            try:
                # Ironically, `np.broadcast_arrays` seems to be more efficient than
                # `np.broadcast_shapes`
                broadcast_input_shape = np.broadcast_arrays(x0, x1)[0].shape
            except ValueError as v:
                raise ValueError(
                    f"The input arrays `x0` and `x1` with shapes {x0.shape} and "
                    f"{x1.shape} can not be broadcast to a common shape."
                ) from v

        return x0, x1, broadcast_input_shape
