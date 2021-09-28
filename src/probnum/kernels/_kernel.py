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
    ...         super().__init__(input_dim=1, output_dim=None)
    ...
    ...     def _evaluate(self, x0, x1=None):
    ...         return self._euclidean_inner_products(x0, x1) + self.constant

    We can now evaluate the kernel like so.

    >>> import numpy as np
    >>> k = CustomLinearKernel(constant=1.0)
    >>> xs = np.linspace(0, 1, 4)[:, None]
    >>> k.matrix(xs)
    array([[1.        , 1.        , 1.        , 1.        ],
           [1.        , 1.11111111, 1.22222222, 1.33333333],
           [1.        , 1.22222222, 1.44444444, 1.66666667],
           [1.        , 1.33333333, 1.66666667, 2.        ]])
    """

    def __init__(
        self,
        input_dim: IntArgType,
        output_dim: Optional[IntArgType] = None,
    ):
        self._input_dim = int(input_dim)

        self._output_dim = None

        if output_dim is not None:
            self._output_dim = int(output_dim)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __call__(
        self,
        x0: _InputType,
        x1: Optional[_InputType],
    ) -> np.ndarray:
        """Evaluate the kernel.

        The inputs are broadcast to a common shape following the "kernel broadcasting"
        rules outlined in the "Notes" section.

        Parameters
        ----------
        x0 :
            An array of shape ``()`` or ``(Nn, ..., N2, N1, D_in)``, where ``D_in`` is
            either ``1`` or :attr:`input_dim`, whose entries will be passed to the first
            argument of the kernel.
        x1 :
            An array of shape ``()`` or ``(Mm, ..., M2, M1, D_in)``, where ``D_in`` is
            either ``1`` or :attr:`input_dim`, whose entries will be
            passed to the second argument of the kernel. Can also be set to ``None``,
            in which case the function will be have as if ``x1 = x0``.

        Returns
        -------
        k_x0_x1 :
            The kernel evaluated at ``x0`` and ``x1``.
            If :attr:`output_dim` is ``None``, this method returns array of shape
            ``(Lk, ..., L2, L1)`` whose entry at index ``(ik, ..., i2, i1)`` contains
            the kernel evaluation ``k(x0[ik, ..., i2, i1, :], x1[il, ..., i2, i1, :])``,
            while for a positive integer :attr:`output_dim`, it returns an array of
            shape ``(output_dim, output_dim, Lk, ..., L2, L1)`` whose entry at index
            ``(j, l, ik, ..., i2, i1)`` contains the evaluation
            ``k[j, l](x0[ik, ..., i2, i1, :], x1[il, ..., i2, i1, :])`` of the
            (cross-)covariance ``k[j, l]`` between outputs `j` and `l`
            (assuming that ``x0`` and ``x1`` have been broadcast according to the rules
            described in the "Notes" section).

        Raises
        ------
        TODO

        See Also
        --------
        matrix: TODO
        diag: TODO

        Notes
        -----
        A :class:`Kernel` operates on its two inputs by a slightly modified version of
        Numpy's broadcasting rules. First of all, the operation of the kernel is
        vectorized over all but the last dimension, applying standard broadcasting
        rules. An input with shape ``()`` is promoted to an input with shape ``(1,)``.
        Additionally, a `1` along the last axis of an input is interpreted as a (set of)
        point(s) with equal coordinates in all input dimensions. We refer to this
        modified set of broadcasting rules as "kernel broadcasting".

        Examples
        --------
        See class docstring: :class:`Kernel`.
        """

        x0, x1, _, output_shape = self._kernel_broadcasting(x0, x1)

        k_x0_x1 = self._evaluate(x0, x1)

        assert k_x0_x1.shape == output_shape

        return k_x0_x1

    def matrix(
        self,
        x0: np.ndarray,
        x1: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x0 = np.array(x0)
        x1 = x0 if x1 is None else np.array(x1)

        # Shape checking
        errmsg = (
            "`{argname}` must have shape `(N, D)` or `(D,)`, where `D` is the input "
            f"dimension of the kernel (D = {self.input_dim}), but an array with shape "
            "`{shape}` was given."
        )

        if not (1 <= x0.ndim <= 2 and x0.shape[-1] == self.input_dim):
            raise ValueError(errmsg.format(argname="x0", shape=x0.shape))

        if not (1 <= x1.ndim <= 2 and x1.shape[-1] == self.input_dim):
            raise ValueError(errmsg.format(argname="x1", shape=x1.shape))

        # Pairwise kernel evaluation
        return self(x0[:, None, :], x1[None, :, :])

    def diag(self, x):
        pass

    @property
    def input_dim(self) -> int:
        """Dimension of arguments of the covariance function.

        The dimension of inputs to the covariance function :math:`k : \\mathbb{R}^{
        d_{in}} \\times \\mathbb{R}^{d_{in}} \\rightarrow
        \\mathbb{R}^{d_{out} \\times d_{out}}`.
        """
        return self._input_dim

    @property
    def output_dim(self) -> Optional[int]:
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

    def _kernel_broadcasting(
        self,
        x0: _InputType,
        x1: Optional[_InputType] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], ShapeType, ShapeType]:
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

        output_shape = broadcast_input_shape[:-1]

        if self.output_dim is not None:
            output_shape = 2 * (self.output_dim,) + output_shape

        return x0, x1, broadcast_input_shape, output_shape

    def _euclidean_inner_products(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> np.ndarray:
        """Implementation of the Euclidean inner product, which supports kernel
        broadcasting semantics."""
        prods = x0 ** 2 if x1 is None else x0 * x1

        if prods.shape[-1] == 1:
            return self.input_dim * prods[..., 0]

        return np.sum(prods, axis=-1)


class IsotropicMixin:
    r"""Mixin for isotropic kernels.

    An isotropic kernel is a kernel which only depends on the Euclidean norm of the
    distance between the arguments, i.e.

    .. math ::

        k(x_0, x_1) = k(\lVert x_0 - x_1 \rVert_2).

    Hence, all isotropic kernels are stationary.
    """

    def _squared_euclidean_distances(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> np.ndarray:
        """Implementation of the squared Euclidean distance, which supports kernel
        broadcasting semantics."""
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[:-1],
            )

        sqdiffs = (x0 - x1) ** 2

        if sqdiffs.shape[-1] == 1:
            return self.input_dim * sqdiffs[..., 0]

        return np.sum(sqdiffs, axis=-1)

    def _euclidean_distances(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> np.ndarray:
        """Implementation of the Euclidean distance, which supports kernel
        broadcasting semantics."""
        sqdists = self._squared_euclidean_distances(x0, x1)

        if x1 is None:
            return sqdists

        return np.sqrt(sqdists)
