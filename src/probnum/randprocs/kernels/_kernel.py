"""Kernel / covariance function."""

import abc
from typing import Optional, Union

import numpy as np

from probnum import utils as _pn_utils
from probnum.typing import ArrayLike, IntLike, ShapeLike, ShapeType


class Kernel(abc.ABC):
    r"""(Cross-)covariance function(s)

    Abstract base class representing one or multiple (cross-)covariance function(s),
    also known as kernels.
    A cross-covariance function

    .. math::
        :nowrap:

        \begin{equation}
            k_{fg} \colon
            \mathcal{X}^{d_\text{in}} \times \mathcal{X}^{d_\text{in}}
            \to \mathbb{R}
        \end{equation}

    is a function of two arguments :math:`x_0` and :math:`x_1`, which represents the
    covariance between two evaluations :math:`f(x_0)` and :math:`g(x_1)` of two
    scalar-valued random processes :math:`f` and :math:`g` (or, equivalently, two
    outputs :math:`h_i(x_0)` and :math:`h_j(x_1)` of a vector-valued random process
    :math:`h`).
    If :math:`f = g`, then the cross-covariance function is also referred to as a
    covariance function, in which case it must be symmetric and positive
    (semi-)definite.

    An instance of a :class:`Kernel` can compute multiple different (cross-)covariance
    functions on the same pair of inputs simultaneously. For instance, it can be used to
    compute the full covariance matrix

    .. math::
        :nowrap:

        \begin{equation}
            C^f \colon
            \mathcal{X}^{d_\text{in}} \times \mathcal{X}^{d_\text{in}}
            \to \mathbb{R}^{d_\text{out} \times d_\text{out}},
            C^f_{i j}(x_0, x_1) := k_{f_i f_j}(x_0, x_1)
        \end{equation}

    of the vector-valued random process :math:`f`. To this end, we understand any
    :class:`Kernel` with a non-empty :attr:`shape` as a tensor with the given
    :attr:`shape`, which contains different (cross-)covariance functions as its entries.

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
    shape :
        If ``shape`` is set to ``()``, the :class:`Kernel` instance represents a
        single (cross-)covariance function. Otherwise, i.e. if ``shape`` is non-empty,
        the :class:`Kernel` instance represents a tensor of (cross-)covariance
        functions whose shape is given by ``shape``.

    Examples
    --------

    >>> D = 3
    >>> k = pn.randprocs.kernels.Linear(input_dim=D)

    Generate some input data.

    >>> xs = np.repeat(np.linspace(0, 1, 4)[:, None], D, axis=-1)
    >>> xs.shape
    (4, 3)
    >>> xs
    array([[0.        , 0.        , 0.        ],
           [0.33333333, 0.33333333, 0.33333333],
           [0.66666667, 0.66666667, 0.66666667],
           [1.        , 1.        , 1.        ]])

    We can compute kernel matrices like so.

    >>> k.matrix(xs)
    array([[0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.33333333, 0.66666667, 1.        ],
           [0.        , 0.66666667, 1.33333333, 2.        ],
           [0.        , 1.        , 2.        , 3.        ]])

    Inputs to :meth:`Kernel.__call__` are broadcast according to the "kernel
    broadcasting" rules detailed in the "Notes" section of the :meth:`Kernel._call__`
    documentation.

    >>> k(xs[:, None, :], xs[None, :, :])  # same as `.matrix`
    array([[0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.33333333, 0.66666667, 1.        ],
           [0.        , 0.66666667, 1.33333333, 2.        ],
           [0.        , 1.        , 2.        , 3.        ]])

    A shape of ``1`` along the last axis is broadcast to :attr:`input_dim`.

    >>> xs_d1 = xs[:, [0]]
    >>> xs_d1.shape
    (4, 1)
    >>> xs_d1
    array([[0.        ],
           [0.33333333],
           [0.66666667],
           [1.        ]])
    >>> k(xs_d1[:, None, :], xs_d1[None, :, :])  # same as `.matrix`
    array([[0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.33333333, 0.66666667, 1.        ],
           [0.        , 0.66666667, 1.33333333, 2.        ],
           [0.        , 1.        , 2.        , 3.        ]])
    >>> k(xs[:, None, :], xs_d1[None, :, :])  # same as `.matrix`
    array([[0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.33333333, 0.66666667, 1.        ],
           [0.        , 0.66666667, 1.33333333, 2.        ],
           [0.        , 1.        , 2.        , 3.        ]])

    No broadcasting is applied if both inputs have the same shape. For instance, one can
    efficiently compute just the diagonal of the kernel matrix via

    >>> k(xs, xs)
    array([0.        , 0.33333333, 1.33333333, 3.        ])
    >>> k(xs, None)  # x1 = None is an efficient way to set x1 == x0
    array([0.        , 0.33333333, 1.33333333, 3.        ])

    and the diagonal above the main diagonal of the kernel matrix is retrieved through

    >>> k(xs[:-1, :], xs[1:, :])
    array([0.        , 0.66666667, 2.        ])
    """

    def __init__(
        self,
        input_dim: IntLike,
        shape: ShapeLike = (),
    ):
        self._input_dim = int(input_dim)

        self._shape = _pn_utils.as_shape(shape)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __call__(
        self,
        x0: ArrayLike,
        x1: Optional[ArrayLike],
    ) -> Union[np.ndarray, np.floating]:
        """Evaluate the (cross-)covariance function(s).

        The inputs are broadcast to a common shape following the "kernel broadcasting"
        rules outlined in the "Notes" section.

        Parameters
        ----------
        x0 : array-like
            An array of shape ``()`` or ``(Nn, ..., N2, N1, D_in)``, where ``D_in`` is
            either ``1`` or :attr:`input_dim`, whose entries will be passed to the first
            argument of the kernel.
        x1 : array-like
            An array of shape ``()`` or ``(Mm, ..., M2, M1, D_in)``, where ``D_in`` is
            either ``1`` or :attr:`input_dim`, whose entries will be
            passed to the second argument of the kernel. Can also be set to ``None``,
            in which case the function will behave as if ``x1 = x0``.

        Returns
        -------
        k_x0_x1 : numpy.ndarray
            The (cross-)covariance function(s) evaluated at ``x0`` and ``x1``.
            If :attr:`shape` is ``()``, this method returns an array of shape
            ``(Lk, ..., L2, L1)`` whose entry at index ``(ik, ..., i2, i1)`` contains
            the evaluation of the (cross-)covariance function at the inputs
            ``x0[ik, ..., i2, i1, :] and ``x1[il, ..., i2, i1, :])``.
            For any non-empty :attr:`shape`, it returns an array of shape
            ``(Sl, ..., S2, S1, Lk, ..., L2, L1)``, where ``S`` is :attr:`shape`, whose
            entry at index ``(sl, ..., s2, s1, ik, ..., i2, i1)`` contains
            evaluation of the (cross-)covariance function at index ``(sl, ..., s2, s1)``
            at the inputs ``x0[ik, ..., i2, i1, :]`` and ``x1[ik, ..., i2, i1, :]``.
            Above, we assume that ``x0`` and ``x1`` have been broadcast according to the
            rules described in the "Notes" section.

        Raises
        ------
        ValueError
            If the inputs can not be "kernel broadcast" to a common shape.

        See Also
        --------
        matrix: Convenience function to compute a kernel matrix, i.e. a matrix of
            pairwise evaluations of the kernel on two sets of points.

        Notes
        -----
        A :class:`Kernel` operates on its two inputs by a slightly modified version of
        Numpy's broadcasting rules. First of all, the operation of the kernel is
        vectorized over all but the last dimension, applying standard broadcasting
        rules. An input with shape ``()`` is promoted to an input with shape ``(1,)``.
        Additionally, a ``1`` along the last axis of an input is interpreted as a (set
        of) point(s) with equal coordinates in all input dimensions, i.e. the inputs are
        broadcast to :attr:`input_dim` dimensions along the last axis. We refer to this
        modified set of broadcasting rules as "kernel broadcasting".

        Examples
        --------
        See documentation of class :class:`Kernel`.
        """

        x0 = np.atleast_1d(x0)

        if x1 is not None:
            x1 = np.atleast_1d(x1)

        # Shape checking
        broadcast_input_shape = self._kernel_broadcast_shapes(x0, x1)

        # Evaluate the kernel
        k_x0_x1 = self._evaluate(x0, x1)

        assert k_x0_x1.shape == self._shape + broadcast_input_shape[:-1]

        return k_x0_x1

    def matrix(
        self,
        x0: ArrayLike,
        x1: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """A convenience function for computing a kernel matrix for two sets of inputs.

        This is syntactic sugar for ``k(x0[:, None, :], x1[None, :, :])``. Hence, it
        computes the matrix of pairwise covariances between two sets of input points.
        If ``k`` represents a covariance function, then the resulting matrix will be
        symmetric positive (semi-)definite for ``x0 == x1``.

        Parameters
        ----------
        x0 : array-like
            First set of inputs to the (cross-)covariance function as an array of shape
            ``(M, D)``, where ``D`` is either 1 or :attr:`input_dim`.
        x1 : array-like
            Optional second set of inputs to the (cross-)covariance function as an array
            of shape ``(N, D)``, where ``D`` is either 1 or :attr:`input_dim`.
            If ``x1`` is not specified, the function behaves as if ``x1 = x0``.

        Returns
        -------
        kernmat : numpy.ndarray
            The matrix / stack of matrices containing the pairwise evaluations of the
            (cross-)covariance function(s) on ``x0`` and ``x1`` as an array of shape
            ``(M, N)`` if :attr:`shape` is ``()`` or
            ``(S[l - 1], ..., S[1], S[0], M, N)``, where ``S`` is :attr:`shape` if
            :attr:`shape` is non-empty.

        Raises
        ------
        ValueError
            If the shapes of the inputs don't match the specification.

        See Also
        --------
        __call__: Evaluate the kernel more flexibly.

        Examples
        --------
        See documentation of class :class:`Kernel`.
        """

        x0 = np.array(x0)
        x1 = x0 if x1 is None else np.array(x1)

        # Shape checking
        errmsg = (
            "`{argname}` must have shape `(N, D)` or `(D,)`, where `D` is the input "
            f"dimension of the kernel (D = {self.input_dim}), but an array with shape "
            "`{shape}` was given."
        )

        if not (1 <= x0.ndim <= 2 and x0.shape[-1] in (self.input_dim, 1)):
            raise ValueError(errmsg.format(argname="x0", shape=x0.shape))

        if not (1 <= x1.ndim <= 2 and x1.shape[-1] in (self.input_dim, 1)):
            raise ValueError(errmsg.format(argname="x1", shape=x1.shape))

        # Pairwise kernel evaluation
        return self(x0[:, None, :], x1[None, :, :])

    @property
    def input_dim(self) -> int:
        """Dimension of arguments of the covariance function."""
        return self._input_dim

    @property
    def shape(self) -> ShapeType:
        """If :attr:`shape` is ``()``, the :class:`Kernel` instance represents a
        single (cross-)covariance function. Otherwise, i.e. if :attr:`shape` is
        non-empty, the :class:`Kernel` instance represents a tensor of
        (cross-)covariance functions whose shape is given by ``shape``."""
        return self._shape

    @abc.abstractmethod
    def _evaluate(
        self,
        x0: ArrayLike,
        x1: Optional[ArrayLike],
    ) -> Union[np.ndarray, np.float_]:
        """Implementation of the kernel evaluation which is called after input checking.

        When implementing a particular kernel, the subclass should implement the kernel
        computation by overwriting this method. It is called by the
        :meth:`Kernel.__call__` method after applying input checking and promoting
        scalar inputs, i.e. inputs with shape ``()``, to arrays with shape ``(1,)``.
        The implementation of the kernel must implement the rules of kernel broadcasting
        which are outlined in the "Notes" section of the :meth:`Kernel.__call__`
        docstring. Note that the inputs are not automatically broadcast to a common
        shape, but it is guaranteed that the inputs can be broadcast to a common shape
        when applying the rules of "kernel broadcasting".

        Parameters
        ----------
        x0 : array-like
            An array of shape ``(Nn, ..., N2, N1, D_in)``, where ``D_in`` is either
            ``1`` or :attr:`input_dim`, whose entries will be passed to the first
            argument of the kernel.
        x1 : array-like
            An array of shape ``(Mm, ..., M2, M1, D_in)``, where ``D_in`` is either
            ``1`` or :attr:`input_dim`, whose entries will be passed to the second
            argument of the kernel. Can also be set to ``None``, in which case the
            method must behave as if ``x1 == x0``.

        Returns
        -------
        k_x0_x1 : numpy.ndarray
            The (cross-)covariance function(s) evaluated at ``x0`` and ``x1`` as an
            array of shape ``(S[l - 1], ..., S[1], S[0], Lk, ..., L2, L1)``,
            where ``S`` is :attr:`shape`, whose entry at index
            ``(sl, ..., s2, s1, ik, ..., i2, i1)`` contains the
            evaluation of the (cross-)covariance function at index ``(sl, ..., s2, s1)``
            at the inputs ``x0[ik, ..., i2, i1, :]`` and ``x1[ik, ..., i2, i1, :]``,
            where we assume that ``x0`` and ``x1`` have been broadcast according to the
            rules described in the "Notes" section of :meth:`Kernel.__call__`.
        """

    def _kernel_broadcast_shapes(
        self,
        x0: np.ndarray,
        x1: Optional[np.ndarray] = None,
    ) -> ShapeType:
        """Applies the "kernel broadcasting" rules to the input shapes.

        A description of the "kernel broadcasting" rules can be found in the docstring
        of :meth:`Kernel.__call__`.
        Appropriate exceptions are raised if the shapes can not be "kernel broadcast" to
        a common shape.

        Parameters
        ----------
        x0 :
            First input to the covariance function.
        x1 :
            Second input to the covariance function.

        Returns
        -------
        broadcast_input_shape : tuple of int
            Shape of the inputs after applying "kernel broadcasting".

        Raises
        -------
        ValueError
            If the inputs can not be "kernel broadcast" to a common shape.
        """

        err_msg = (
            "`{argname}` can not be broadcast to the kernel's input dimension "
            f"{self.input_dim} along the last axis, since "
            "`{argname}.shape = {shape}`."
        )

        if x0.shape[-1] not in (self.input_dim, 1):
            # This will never be called if the original input was a scalar
            raise ValueError(err_msg.format(argname="x0", shape=x0.shape))

        broadcast_input_shape = x0.shape

        if x1 is not None:
            if x1.shape[-1] not in (self.input_dim, 1):
                # This will never be called if the original input was a scalar
                raise ValueError(err_msg.format(argname="x1", shape=x1.shape))

            try:
                # Ironically, `np.broadcast_arrays` seems to be more efficient than
                # `np.broadcast_shapes`
                broadcast_input_shape = np.broadcast_arrays(x0, x1)[0].shape
            except ValueError as v:
                raise ValueError(
                    f"The input arrays `x0` and `x1` with shapes {x0.shape} and "
                    f"{x1.shape} can not be broadcast to a common shape."
                ) from v

        if broadcast_input_shape[-1] == 1:
            broadcast_input_shape = broadcast_input_shape[:-1] + (self.input_dim,)

        return broadcast_input_shape

    def _euclidean_inner_products(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> np.ndarray:
        """Implementation of the Euclidean inner product, which supports kernel
        broadcasting semantics."""
        prods = x0 ** 2 if x1 is None else x0 * x1

        if prods.shape[-1] == 1:
            return self.input_dim * prods[..., 0]

        return np.sum(prods, axis=-1)


class IsotropicMixin(abc.ABC):  # pylint: disable=too-few-public-methods
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
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[:-1],
            )

        return np.sqrt(self._squared_euclidean_distances(x0, x1))
