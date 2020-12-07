"""Kernel / covariance function."""

from typing import Callable, Generic, Optional, Tuple, TypeVar, Union

import numpy as np
import scipy.spatial

import probnum.utils as _utils
from probnum.type import IntArgType, ScalarArgType, ShapeArgType

_InputType = TypeVar("InputType")


class Kernel(Generic[_InputType]):
    """Kernel / covariance function.

    Kernels are a generalization of a positive-definite function or matrix. They
    typically describe the covariance function of a random process and thus describe
    its spatial or temporal variation. If evaluated at two sets of points a kernel
    gives the covariance of the random process at these locations.

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
    output_dim :
        Output dimension of the kernel.
    kernelfun :
        Function defining the kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.kernels import Kernel
    >>> # Data
    >>> x = np.array([[1, 2], [-1, -1]])
    >>> # Custom kernel from a (non-vectorized) covariance function
    >>> k = Kernel(kernelfun=lambda x0, x1: (x0.T @ x1 - 1.0) ** 2, input_dim=2)
    >>> k(x)
    array([[16., 16.],
           [16.,  1.]])
    """

    # pylint: disable="invalid-name"
    def __init__(
        self,
        input_dim: IntArgType,
        output_dim: IntArgType = 1,
        kernelfun: Optional[
            Callable[[_InputType, Optional[_InputType]], np.ndarray]
        ] = None,
    ):
        self.__input_dim = np.int_(_utils.as_numpy_scalar(input_dim))
        self.__output_dim = np.int_(_utils.as_numpy_scalar(output_dim))
        self.__kernelfun = (
            self._as_vectorized_kernel_function(fun=kernelfun)
            if kernelfun is not None
            else None
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

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
        if self.__kernelfun is not None:
            x0, x1, _ = self._check_and_transform_input(x0=x0, x1=x1)
            return self._transform_kernelmatrix(
                kerneval=self.__kernelfun(x0, x1), x0_shape=x0.shape, x1_shape=x1.shape
            )
        else:
            raise NotImplementedError

    @property
    def input_dim(self) -> int:
        """Dimension of arguments of the covariance function.

        The dimension of inputs to the covariance function :math:`k : \\mathbb{R}^{
        d_{in}} \\times \\mathbb{R}^{d_{in}} \\rightarrow
        \\mathbb{R}^{d_{out} \\times d_{out}}`.
        """
        return self.__input_dim

    @property
    def output_dim(self) -> int:
        """Dimension of the evaluated covariance function.

        The resulting evaluated kernel :math:`k(x_0, x_1) \\in
        \\mathbb{R}^{d_{out} \\times d_{out}}` has *shape=(output_dim,
        output_dim)*.
        """
        return self.__output_dim

    def _check_and_transform_input(
        self,
        x0: _InputType,
        x1: Optional[_InputType] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Transform inputs to the kernel matrix.

        Transforms inputs into :class:`numpy.ndarray` and standardizes their shape.

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
        equal_inputs :
            Are the two inputs the same?

        Raises
        -------
        ValueError :
            If input shapes of x0 and x1 do not match the kernel input dimension or
            each other.
        """
        # pylint: disable="too-many-boolean-expressions"
        # Transform into array and add second argument
        x0 = np.asarray(x0)
        equal_inputs = False
        if x1 is None:
            x1 = x0
            equal_inputs = True
        else:
            x1 = np.asarray(x1)

        # Error message
        err_msg = (
            f"Argument shapes x0.shape={x0.shape} and x1.shape="
            f"{x1.shape} do not match kernel input dimension "
            f"{self.input_dim}. Try passing either two vectors or two matrices with the "
            f"second dimension equalling the kernel input dimension."
        )

        # Check and promote shapes
        if x1 is None:

            if (
                (x0.ndim == 0 and self.input_dim > 1)  # Scalar input
                or (x0.ndim == 1 and x0.shape[0] != self.input_dim)  # Vector input
                or (x0.ndim >= 2 and x0.shape[1] != self.input_dim)  # Matrix input
            ):
                raise ValueError(err_msg)
            equal_inputs = True
        else:
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

        return x0, x1, equal_inputs

    def _transform_kernelmatrix(
        self,
        kerneval: Union[np.ndarray, ScalarArgType],
        x0_shape: ShapeArgType,
        x1_shape: ShapeArgType,
    ) -> Union[np.float_, np.ndarray]:
        """Transform the kernel matrix based on the given arguments.

        Standardizes the given evaluation of the covariance function to the correct
        shape determined by the input arguments. If inputs are vectors the
        output is a numpy scalar if the output dimension of the kernel is 1,
        otherwise *shape=(output_dim, output_dim)*. If inputs represent multiple
        observations, then the resulting matrix has *shape=(n0, n1) or
        (n0, n1, output_dim, output_dim)*.

        Parameters
        ----------
        kerneval
            Covariance function evaluated at ``x0`` and ``x1``.
        x0_shape :
            Shape of the first input to the covariance function.
        x1_shape :
            Shape of the second input to the covariance function.
        """
        if len(x0_shape) <= 1 and len(x1_shape) <= 1:
            if self.output_dim == 1:
                return _utils.as_numpy_scalar(kerneval.squeeze())
            else:
                kern_shape = ()
        else:
            kern_shape = (x0_shape[0], x1_shape[0])

        if self.output_dim > 1:
            kern_shape += (
                self.output_dim,
                self.output_dim,
            )

        return kerneval.reshape(kern_shape)

    def _as_vectorized_kernel_function(
        self,
        fun: Callable[
            [_InputType, Optional[_InputType]], Union[np.ndarray, ScalarArgType]
        ],
    ) -> Callable[[_InputType, Optional[_InputType]], Union[np.ndarray, np.float_]]:
        """Transform a function into a vectorized covariance function.

        Creates a kernel / covariance function from a (non-vectorized) function
        :math:`k : \\mathbb{R}^{d_{in}} \\times \\mathbb{R}^{d_{in}} \\rightarrow
        \\mathbb{R}`. When a kernel matrix is computed the given function ``fun`` is
        applied to all pairs of inputs.

        Parameters
        ----------
        fun
            (Non-vectorized) covariance function.
        """
        if callable(fun):
            rng = np.random.default_rng(42)
            x0 = rng.normal(size=(2, self.input_dim))
            x1 = rng.normal(size=(3, self.input_dim))
            # Check if given function is already vectorized
            try:
                kernmat = fun(x0, x1)
                outshape = (x0.shape[0], x1.shape[0])
                if (self.output_dim == 1 and kernmat.shape == outshape) or (
                    self.output_dim > 1
                    and kernmat.shape == (outshape + (self.output_dim, self.output_dim))
                ):
                    # Make second argument optional
                    def _fun(x0, x1=None):
                        if x1 is None:
                            x1 = x0
                        return fun(x0, x1)

                    return _fun
            except (AttributeError, ValueError):
                pass

            # Vectorize given kernel function
            def _kernfun_vectorized(x0, x1=None) -> np.ndarray:
                # pylint: disable=invalid-name
                x0 = np.asarray(x0)
                if x1 is None:
                    x1 = x0
                else:
                    x1 = np.asarray(x1)
                if x0.ndim < 2 and x1.ndim < 2:
                    return np.asarray(fun(x0, x1)).reshape(1, 1)
                else:
                    x0 = np.atleast_2d(x0)
                    x1 = np.atleast_2d(x1)

                    # Evaluate fun pairwise for all rows of x0 and x1
                    return scipy.spatial.distance.cdist(x0, x1, metric=fun)

            return _kernfun_vectorized
        else:
            raise TypeError(
                f"The given covariance function of type {type(fun)} is "
                f"not a callable."
            )
