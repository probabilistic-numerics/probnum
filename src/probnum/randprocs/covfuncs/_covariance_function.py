"""Abstract base class for (cross-)covariance functions."""

from __future__ import annotations

import abc
import functools
import operator
from typing import Optional, Union

import numpy as np

from probnum import linops, utils as _pn_utils
from probnum.typing import ArrayLike, ScalarLike, ShapeLike, ShapeType

BinaryOperandType = Union["CovarianceFunction", ScalarLike]


class CovarianceFunction(abc.ABC):
    r"""(Cross-)covariance function.

    A cross-covariance function

    .. math::
        :nowrap:

        \begin{equation}
            k \colon
            \mathbb{X}_0 \times \mathbb{X}_1
            \to \mathbb{R}^{d^\text{out}_0 \times d^\text{out}_1},
            (x_0, x_1) \mapsto \operatorname{Cov}[f_0(x_0), f_1(x_1)]
        \end{equation}

    is a function of two arguments :math:`x_0 \in \mathbb{X}_0` and :math:`x_1 \in \
    \mathbb{X}_1` (often :math:`\mathbb{X}_i \subset \mathbb{R}^{d^\text{in}_i}`), whose
    output corresponds to the covariance (matrix) between two evaluations
    :math:`f_0(x_0) \in \mathbb{R}^{d^\text{out}_0}` and :math:`f_1(x_1) \in \
    \mathbb{R}^{d^\text{out}_1}` of two (vector-valued) :class:`~probnum.randprocs.\
    RandomProcess`\ es :math:`f_0` and :math:`f_1`.
    If :math:`f_0 = f_1`, then a cross-covariance function is also referred to as a
    covariance function or a kernel, in which case it must be symmetric and positive
    (semi-)definite.

    Parameters
    ----------
    input_shape_0
        :attr:`~probnum.randprocs.RandomProcess.input_shape` of the :class:`~probnum.\
        randprocs.RandomProcess` :math:`f_0`.
        This defines the shape of the :class:`CovarianceFunction`\ 's first input
        :math:`x_0`.
    input_shape_1
        :attr:`~probnum.randprocs.RandomProcess.input_shape` of the :class:`~probnum.\
        randprocs.RandomProcess` :math:`f_1`.
        This defines the shape of the :class:`CovarianceFunction`\ 's second input
        :math:`x_1`.
    input_shape
        Convenience argument, which can be used to set ``input_shape_0 == input_shape_1
        == input_shape``.
        If ``input_shape`` is specified, then ``input_shape_{0,1}`` must be set to
        :data:`None`.
    output_shape_0
        :attr:`~probnum.randprocs.RandomProcess.output_shape` of the
        :class:`~probnum.randprocs.RandomProcess` :math:`f_0`.
    output_shape_1
        :attr:`~probnum.randprocs.RandomProcess.output_shape` of the
        :class:`~probnum.randprocs.RandomProcess` :math:`f_1`.

    Examples
    --------

    >>> from probnum.randprocs.covfuncs import Linear
    >>> D = 3
    >>> k = Linear(input_shape=D)
    >>> k.input_shape_0
    (3,)
    >>> k.input_shape_1
    (3,)
    >>> k.output_shape_0
    ()
    >>> k.output_shape_1
    ()

    Generate some input data.

    >>> N = 4
    >>> xs = np.linspace(0, 1, N * D).reshape(N, D)
    >>> xs.shape
    (4, 3)
    >>> xs
    array([[0.        , 0.09090909, 0.18181818],
           [0.27272727, 0.36363636, 0.45454545],
           [0.54545455, 0.63636364, 0.72727273],
           [0.81818182, 0.90909091, 1.        ]])

    We can compute covariance matrices of multiple evaluations like so.

    >>> k.matrix(xs)
    array([[0.04132231, 0.11570248, 0.19008264, 0.26446281],
           [0.11570248, 0.41322314, 0.7107438 , 1.00826446],
           [0.19008264, 0.7107438 , 1.23140496, 1.75206612],
           [0.26446281, 1.00826446, 1.75206612, 2.49586777]])

    The :meth:`__call__` method is vectorized over the "batch shapes" of the inputs,
    applying standard NumPy broadcasting.

    >>> k(xs[:, None], xs[None, :])  # same as `.matrix`
    array([[0.04132231, 0.11570248, 0.19008264, 0.26446281],
           [0.11570248, 0.41322314, 0.7107438 , 1.00826446],
           [0.19008264, 0.7107438 , 1.23140496, 1.75206612],
           [0.26446281, 1.00826446, 1.75206612, 2.49586777]])

    No broadcasting is applied if both inputs have the same shape. For instance, one can
    efficiently compute the marginal variance of a set of data points via

    >>> k(xs, xs)
    array([0.04132231, 0.41322314, 1.23140496, 2.49586777])
    >>> k(xs, None)  # x1 = None is an efficient way to set x1 == x0
    array([0.04132231, 0.41322314, 1.23140496, 2.49586777])

    :class:`CovarianceFunction`\ s support basic arithmetic operations. For example, we
    can model independent measurement noise as follows:

    >>> from probnum.randprocs.covfuncs import WhiteNoise
    >>> k_noise = k + 0.1 * WhiteNoise(input_shape=D)
    >>> k_noise.matrix(xs)
    array([[0.14132231, 0.11570248, 0.19008264, 0.26446281],
           [0.11570248, 0.51322314, 0.7107438 , 1.00826446],
           [0.19008264, 0.7107438 , 1.33140496, 1.75206612],
           [0.26446281, 1.00826446, 1.75206612, 2.59586777]])
    """

    def __init__(
        self,
        *,
        input_shape_0: Optional[ShapeLike] = None,
        input_shape_1: Optional[ShapeLike] = None,
        input_shape: Optional[ShapeLike] = None,
        output_shape_0: ShapeLike = (),
        output_shape_1: ShapeLike = (),
    ):
        assert (
            input_shape_0 is not None and input_shape_1 is not None
        ) or input_shape is not None, (
            "Either `input_shape_0` and `input_shape_1` or `input_shape` must be given."
        )

        if input_shape is not None:
            assert input_shape_0 is None and input_shape_1 is None, (
                "If `input_shape` is given, `input_shape_0` and `input_shape_1` must "
                "be set to `None`."
            )

            input_shape_0 = input_shape
            input_shape_1 = input_shape

        self._input_shape_0 = _pn_utils.as_shape(input_shape_0)
        self._input_shape_1 = _pn_utils.as_shape(input_shape_1)

        self._output_shape_0 = _pn_utils.as_shape(output_shape_0)
        self._output_shape_1 = _pn_utils.as_shape(output_shape_1)

    @property
    def input_shape_0(self) -> ShapeType:
        r""":attr:`~probnum.randprocs.RandomProcess.input_shape` of the
        :class:`~probnum.randprocs.RandomProcess` :math:`f_0`.
        This defines the shape of a single, i.e. non-batched, first argument :math:`x_0`
        of the :class:`CovarianceFunction`."""
        return self._input_shape_0

    @property
    def input_ndim_0(self) -> int:
        r"""Syntactic sugar for ``len(``\ :attr:`input_shape_0`\ ``)``."""
        return len(self.input_shape_0)

    @functools.cached_property
    def input_size_0(self) -> int:
        """Syntactic sugar for the product of all entries in :attr:`input_shape_0`."""
        return functools.reduce(operator.mul, self.input_shape_0, 1)

    @property
    def input_shape_1(self) -> ShapeType:
        r""":attr:`~probnum.randprocs.RandomProcess.input_shape` of the
        :class:`~probnum.randprocs.RandomProcess` :math:`f_1`.
        This defines the shape of a single, i.e. non-batched, second argument
        :math:`x_1` of the :class:`CovarianceFunction`."""
        return self._input_shape_1

    @property
    def input_ndim_1(self) -> int:
        r"""Syntactic sugar for ``len(``\ :attr:`input_shape_1`\ ``)``."""
        return len(self.input_shape_1)

    @functools.cached_property
    def input_size_1(self) -> int:
        """Syntactic sugar for the product of all entries in :attr:`input_shape_1`."""
        return functools.reduce(operator.mul, self.input_shape_1, 1)

    @property
    def input_shape(self) -> ShapeType:
        r"""Shorthand for the input shape of a covariance function with
        :attr:`input_shape_0` ``==`` :attr:`input_shape_1`.

        Raises
        ------
        ValueError
            If the input shapes of the :class:`CovarianceFunction` are not equal.
        """
        if self.input_shape_0 != self.input_shape_1:
            raise ValueError(
                "The input shapes of the `CovarianceFunction` are not equal."
            )

        return self.input_shape_0

    @property
    def input_ndim(self) -> int:
        r"""Syntactic sugar for ``len(``\ :attr:`input_shape`\ ``)``."""
        return len(self.input_shape)

    @functools.cached_property
    def input_size(self) -> int:
        """Syntactic sugar for the product of all entries in :attr:`input_shape`."""
        return functools.reduce(operator.mul, self.input_shape, 1)

    @property
    def output_shape_0(self) -> ShapeType:
        """:attr:`~probnum.randprocs.RandomProcess.output_shape` of the
        :class:`~probnum.randprocs.RandomProcess` :math:`f_0`.

        This defines the first part of the shape of a single, i.e. non-batched, return
        value of :meth:`__call__`.
        """
        return self._output_shape_0

    @property
    def output_ndim_0(self) -> int:
        r"""Syntactic sugar for ``len(``\ :attr:`output_shape_0`\ ``)``."""
        return len(self.output_shape_0)

    @functools.cached_property
    def output_size_0(self) -> int:
        """Syntactic sugar for the product of all entries in :attr:`output_shape_0`."""
        return functools.reduce(operator.mul, self.output_shape_0, 1)

    @property
    def output_shape_1(self) -> ShapeType:
        """:attr:`~probnum.randprocs.RandomProcess.output_shape` of the
        :class:`~probnum.randprocs.RandomProcess` :math:`f_1`.

        This defines the second part of the shape of a single, i.e. non-batched, return
        value of :meth:`__call__`.
        """
        return self._output_shape_1

    @property
    def output_ndim_1(self) -> int:
        r"""Syntactic sugar for ``len(``\ :attr:`output_shape_1`\ ``)``."""
        return len(self.output_shape_1)

    @functools.cached_property
    def output_size_1(self) -> int:
        """Syntactic sugar for the product of all entries in :attr:`output_shape_1`."""
        return functools.reduce(operator.mul, self.output_shape_1, 1)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with"
            f" input_shape_0={self.input_shape_0},"
            f" input_shape_1={self.input_shape_1},"
            f" output_shape_0={self.output_shape_0}, and"
            f" output_shape_1={self.output_shape_1}>"
        )

    def __call__(
        self,
        x0: ArrayLike,
        x1: Optional[ArrayLike],
    ) -> np.ndarray:
        """Evaluate the (cross-)covariance function.

        The evaluation of the (cross-covariance) function is vectorized over the batch
        shapes of the arguments, applying standard NumPy broadcasting.

        Parameters
        ----------
        x0
            *shape=* ``batch_shape_0 +`` :attr:`input_shape_0` -- (Batch of) input(s)
            for the first argument of the :class:`CovarianceFunction`.
        x1
            *shape=* ``batch_shape_1 +`` :attr:`input_shape_1` -- (Batch of) input(s)
            for the second argument of the :class:`CovarianceFunction`.
            Can also be set to ``None``, in which case the function will behave as if
            ``x1 = x0`` (but it is implemented more efficiently).

        Returns
        -------
        k_x0_x1 :
            *shape=* ``bcast_batch_shape +`` :attr:`output_shape_0` ``+``
            :attr:`output_shape_1` --
            The (cross-)covariance function evaluated at ``(x0, x1)``.
            Since the function is vectorized over the batch shapes of the inputs, the
            output array contains the following entries:

            .. code-block:: python

                k_x0_x1[batch_idx] = k(x0[batch_idx, ...], x1[batch_idx, ...])

            where we assume that the batch shapes of ``x0`` and ``x1`` have been
            broadcast to a common shape ``bcast_batch_shape``, and where ``batch_idx``
            is an index compatible with ``bcast_batch_shape``.

        Raises
        ------
        ValueError
            If the shape of :math:`x_0` is not of the form ``batch_shape_0 +``
            :attr:`input_shape_0`.
        ValueError
            If the shape of :math:`x_1` is not of the form ``batch_shape_1 +``
            :attr:`input_shape_1`.
        ValueError
            If the inputs can not be broadcast to a common shape.

        See Also
        --------
        matrix: Convenience function computing the full covariance matrix of evaluations
            at two given sets of input points.

        Examples
        --------
        See documentation of class :class:`CovarianceFunction`.
        """

        x0 = np.asarray(x0)

        if x1 is not None:
            x1 = np.asarray(x1)

        # Shape checking
        broadcast_batch_shape = self._check_shapes(
            x0.shape, x1.shape if x1 is not None else None
        )

        # Evaluate the covariance function
        k_x0_x1 = self._evaluate(x0, x1)

        assert (
            k_x0_x1.shape
            == broadcast_batch_shape + self._output_shape_0 + self._output_shape_1
        )

        return k_x0_x1

    def matrix(
        self,
        x0: ArrayLike,
        x1: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        r"""Matrix containing the pairwise covariances of evaluations of :math:`f_0` and
        :math:`f_1` at the given input points.

        Parameters
        ----------
        x0
            *shape=* ``batch_shape_0 +`` :attr:`input_shape_0` -- (Batch of) input(s)
            for the first argument of the :class:`CovarianceFunction`.
        x1
            *shape=* ``batch_shape_1 +`` :attr:`input_shape_1` -- (Batch of) input(s)
            for the second argument of the :class:`CovarianceFunction`.
            Can also be set to :data:`None`, in which case the function will behave as
            if ``x1 == x0`` (potentially using a more efficient implementation for this
            particular case).

        Returns
        -------
        k_x0_x1
            *shape=* ``(``\ :attr:`output_size_0` ``* N0,`` :attr:`output_size_1`
            ``* N1)``
            *with* ``N0 = prod(batch_shape_0)`` and ``N1 = prod(batch_shape_1)`` --
            The covariance matrix corresponding to the given batches of input points.
            The order of the rows and columns of the covariance matrix corresponds to
            the order of entries obtained by flattening :class:`~numpy.ndarray`\ s with
            shapes :attr:`output_shape_0` ``+ batch_shape_0`` and :attr:`output_shape_0`
            ``+ batch_shape_1`` in "C-order".

        Raises
        ------
        ValueError
            If the shape of :math:`x_0` is not of the form ``batch_shape_0 +``
            :attr:`input_shape_0`.
        ValueError
            If the shape of :math:`x_1` is not of the form ``batch_shape_1 +``
            :attr:`input_shape_1`.
        """
        x0 = self._preprocess_linop_input(x0, argnum=0)

        if x1 is not None:
            x1 = self._preprocess_linop_input(x1, argnum=1)

        k_matrix_x0_x1 = self._evaluate_matrix(x0, x1)

        assert isinstance(k_matrix_x0_x1, np.ndarray)
        assert k_matrix_x0_x1.shape == (
            self.output_size_0 * x0.shape[0],
            self.output_size_1 * (x1.shape[0] if x1 is not None else x0.shape[0]),
        )

        return k_matrix_x0_x1

    def linop(
        self,
        x0: ArrayLike,
        x1: Optional[ArrayLike] = None,
    ) -> linops.LinearOperator:
        r""":class:`~probnum.linops.LinearOperator` representing the pairwise
        covariances of evaluations of :math:`f_0` and :math:`f_1` at the given input
        points.

        Representing the resulting covariance matrix as a matrix-free :class:`~probnum.\
        linops.LinearOperator` is often more efficient than a representation as a
        :class:`~numpy.ndarray`, both in terms of memory and computation time,
        particularly when using iterative methods to solve the associated linear
        systems.

        For instance, covariance matrices induced by separable covariance functions
        (e.g. tensor products of covariance functions or separable multi-output kernels)
        can often be represented as :class:`~probnum.linops.KroneckerProduct`\ s of
        smaller covariance matrices and frameworks like :mod:`pykeops<pykeops.numpy>`
        can be used to implement efficient matrix-vector products with covariance
        matrices without needing to construct the entire matrix in memory.

        Parameters
        ----------
        x0
            *shape=* ``batch_shape_0 +`` :attr:`input_shape_0` -- (Batch of) input(s)
            for the first argument of the :class:`CovarianceFunction`.
        x1
            *shape=* ``batch_shape_1 +`` :attr:`input_shape_1` -- (Batch of) input(s)
            for the second argument of the :class:`CovarianceFunction`.
            Can also be set to :data:`None`, in which case the function will behave as
            if ``x1 == x0`` (potentially using a more efficient implementation for this
            particular case).

        Returns
        -------
        k_x0_x1
            *shape=* ``(``\ :attr:`output_size_0` ``* N0,`` :attr:`output_size_1`
            ``* N1)``
            *with* ``N0 = prod(batch_shape_0)`` and ``N1 = prod(batch_shape_1)`` --
            :class:`~probnum.linops.LinearOperator` representing the covariance matrix
            corresponding to the given batches of input points.
            The order of the rows and columns of the covariance matrix corresponds to
            the order of entries obtained by flattening :class:`~numpy.ndarray`\ s with
            shapes :attr:`output_shape_0` ``+ batch_shape_0`` and :attr:`output_shape_0`
            ``+ batch_shape_1`` in "C-order".

        Raises
        ------
        ValueError
            If the shape of :math:`x_0` is not of the form ``batch_shape_0 +``
            :attr:`input_shape_0`.
        ValueError
            If the shape of :math:`x_1` is not of the form ``batch_shape_1 +``
            :attr:`input_shape_1`.
        """
        x0 = self._preprocess_linop_input(x0, argnum=0)

        if x1 is not None:
            x1 = self._preprocess_linop_input(x1, argnum=1)

        k_linop_x0_x1 = self._evaluate_linop(x0, x1)

        assert isinstance(k_linop_x0_x1, linops.LinearOperator)
        assert k_linop_x0_x1.shape == (
            self.output_size_0 * x0.shape[0],
            self.output_size_1 * (x1.shape[0] if x1 is not None else x0.shape[0]),
        )

        return k_linop_x0_x1

    @abc.abstractmethod
    def _evaluate(
        self,
        x0: np.ndarray,
        x1: Optional[np.ndarray],
    ) -> np.ndarray:
        """Implementation of the covariance function evaluation which is called after
        input checking.

        When implementing a particular covariance function, the subclass should
        overwrite this method.
        It is called by the :meth:`__call__` method after applying input checking.
        The implementation must return the array described in the "Returns" section of
        the :meth:`__call__` method.
        Note that the inputs are not automatically broadcast to a common shape, but it
        is guaranteed that this is possible.

        Parameters
        ----------
        x0
            See argument ``x0`` in the docstring of :meth:`__call__`.
        x1
            See argument ``x1`` in the docstring of :meth:`__call__`.

        Returns
        -------
        k_x0_x1 :
            See "Returns" section in the docstring of :meth:`__call__`.
        """

    def _evaluate_matrix(
        self,
        x0: np.ndarray,
        x1: Optional[np.ndarray],
    ) -> linops.LinearOperator:
        assert x0.ndim == 1 + self.input_ndim_0
        assert x1 is None or x1.ndim == 1 + self.input_ndim_1

        k_x0_x1 = self(x0[:, None, ...], (x1 if x1 is not None else x0)[None, :, ...])

        assert k_x0_x1.ndim == 2 + self.output_ndim_0 + self.output_ndim_1

        batch_shape = k_x0_x1.shape[:2]

        assert k_x0_x1.shape == batch_shape + self.output_shape_0 + self.output_shape_1

        cov_x0_x1 = np.moveaxis(k_x0_x1, 1, -1)
        cov_x0_x1 = np.moveaxis(cov_x0_x1, 0, self.output_ndim_0)

        assert cov_x0_x1.shape == self.output_shape_0 + (
            batch_shape[0],
        ) + self.output_shape_1 + (batch_shape[1],)

        return cov_x0_x1.reshape(
            (
                self.output_size_0 * batch_shape[0],
                self.output_size_1 * batch_shape[1],
            ),
            order="C",
        )

    def _evaluate_linop(
        self,
        x0: np.ndarray,
        x1: Optional[np.ndarray],
    ) -> linops.LinearOperator:
        return linops.Matrix(self._evaluate_matrix(x0, x1))

    def _check_shapes(
        self,
        x0_shape: ShapeType,
        x1_shape: Optional[ShapeType] = None,
    ) -> ShapeType:
        """Checks input argument shapes and computes the broadcast batch shape of both
        inputs.

        This function checks the shapes of the inputs to the :meth:`__call__` method and
        it computes the `bcast_batch_shape` mentioned in the docstring.

        Parameters
        ----------
        x0_shape :
            Shape of the first input to the covariance function.
        x1_shape :
            Shape of the (optional) second input to the covariance function.

        Returns
        -------
        broadcast_batch_shape :
            The `batch_shape` after broadcasting the inputs to a common shape.

        Raises
        -------
        ValueError
            If the shape of :math:`x_0` is not of the form ``batch_shape_0 +``
            :attr:`input_shape_0`.
        ValueError
            If the shape of :math:`x_1` is not of the form ``batch_shape_1 +``
            :attr:`input_shape_1`.
        ValueError
            If the inputs can not be broadcast to a common shape.
        """

        err_msg = (
            "The shape of the input array `x{argnum}` must match "
            "`input_shape_{argnum}`, i.e. `{input_shape}`, along its trailing "
            "dimensions, but an array with shape `{shape}` was given."
        )

        if x0_shape[len(x0_shape) - self.input_ndim_0 :] != self.input_shape_0:
            raise ValueError(
                err_msg.format(argnum=0, input_shape=self.input_shape_0, shape=x0_shape)
            )

        broadcast_batch_shape = x0_shape[: len(x0_shape) - self.input_ndim_0]

        if x1_shape is not None:
            if x1_shape[len(x1_shape) - self.input_ndim_1 :] != self.input_shape_1:
                raise ValueError(
                    err_msg.format(
                        argnum=1, input_shape=self.input_shape_1, shape=x1_shape
                    )
                )

            try:
                broadcast_batch_shape = np.broadcast_shapes(
                    broadcast_batch_shape,
                    x1_shape[: len(x1_shape) - self.input_ndim_1],
                )
            except ValueError as ve:
                err_msg = (
                    f"The input arrays `x0` and `x1` with shapes {x0_shape} and "
                    f"{x1_shape} can not be broadcast to a common shape."
                )
                raise ValueError(err_msg) from ve

        return broadcast_batch_shape

    def _preprocess_linop_input(self, x: ArrayLike, argnum: int) -> np.ndarray:
        x = np.asarray(x)

        assert argnum in (0, 1)

        input_shape = self.input_shape_0 if argnum == 0 else self.input_shape_1
        input_ndim = self.input_ndim_0 if argnum == 0 else self.input_ndim_1

        if not (
            x.ndim >= input_ndim and x.shape[(x.ndim - input_ndim) :] == input_shape
        ):
            raise ValueError(
                f"The shape of `x{argnum}` must must match `input_shape_{argnum}`, "
                f"i.e. `{self.input_shape}`, of the covariance function along its "
                f"trailing dimensions, but an array with shape `{x.shape}` was given."
            )

        return x.reshape((-1,) + self.input_shape, order="C")

    def _euclidean_inner_products(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> np.ndarray:
        """Implementation of the Euclidean inner product, which supports scalar inputs
        and an optional second argument."""
        prods = x0**2 if x1 is None else x0 * x1

        if self.input_ndim == 0:
            return prods

        assert self.input_ndim == 1

        return np.sum(prods, axis=tuple(range(-self.input_ndim, 0)))

    ####################################################################################
    # Binary Arithmetic
    ####################################################################################

    __array_ufunc__ = None
    """
    This prevents numpy from calling elementwise arithmetic operations instead of
    the arithmetic operations defined by `CovarianceFunction`.
    """

    def __add__(self, other: BinaryOperandType) -> CovarianceFunction:
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import add

        return add(self, other)

    def __radd__(self, other: BinaryOperandType) -> CovarianceFunction:
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import add

        return add(other, self)

    def __mul__(self, other: BinaryOperandType) -> CovarianceFunction:
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import mul

        return mul(self, other)

    def __rmul__(self, other: BinaryOperandType) -> CovarianceFunction:
        # pylint: disable=import-outside-toplevel,cyclic-import
        from ._arithmetic import mul

        return mul(other, self)


class IsotropicMixin(abc.ABC):  # pylint: disable=too-few-public-methods
    r"""Mixin for isotropic covariance functions.

    An isotropic covariance function only depends on the norm of the difference of the
    arguments, i.e.

    .. math ::

        k(x_0, x_1) = k(\lVert x_0 - x_1 \rVert).

    Hence, all isotropic covariance functions are stationary.
    """

    def _squared_euclidean_distances(
        self,
        x0: np.ndarray,
        x1: Optional[np.ndarray],
        *,
        scale_factors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Implementation of the squared (modified) Euclidean distance, which supports
        scalar inputs, an optional second argument, and separate scale factors for each
        input dimension."""

        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        sqdiffs = x0 - x1

        if scale_factors is not None:
            sqdiffs *= scale_factors

        sqdiffs *= sqdiffs

        return np.sum(sqdiffs, axis=tuple(range(-self.input_ndim, 0)))

    def _euclidean_distances(
        self,
        x0: np.ndarray,
        x1: Optional[np.ndarray],
        *,
        scale_factors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Implementation of the (modified) Euclidean distance, which supports scalar
        inputs, an optional second argument, and separate scale factors for each input
        dimension."""

        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        return np.sqrt(
            self._squared_euclidean_distances(x0, x1, scale_factors=scale_factors)
        )
