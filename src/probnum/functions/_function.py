"""Function class defining a Callable with fixed in- and output shapes."""

from __future__ import annotations

import abc
import functools
from typing import Callable

import numpy as np

from probnum import utils
from probnum.typing import ArrayLike, ShapeLike, ShapeType


class Function(abc.ABC):
    """Callable with information about the shape of expected in- and outputs.

    This class represents a, uni- or multivariate, scalar- or tensor-valued,
    mathematical function. Hence, the call method should not have any observable
    side-effects.
    Instances of this class can be added and multiplied by a scalar, which means that
    they are elements of a vector space.

    Parameters
    ----------
    input_shape
        Input shape.

    output_shape
        Output shape.

    See Also
    --------
    LambdaFunction : Define a :class:`Function` from an anonymous function.
    ~probnum.functions.Zero : Zero function.
    """

    def __init__(self, input_shape: ShapeLike, output_shape: ShapeLike = ()) -> None:
        self._input_shape = utils.as_shape(input_shape)
        self._input_ndim = len(self._input_shape)

        self._output_shape = utils.as_shape(output_shape)
        self._output_ndim = len(self._output_shape)

    @property
    def input_shape(self) -> ShapeType:
        """Shape of the function's input.

        For a scalar-input function, this is an empty tuple.
        """
        return self._input_shape

    @property
    def input_ndim(self) -> int:
        """Syntactic sugar for ``len(input_shape)``."""
        return self._input_ndim

    @property
    def output_shape(self) -> ShapeType:
        """Shape of the function's output.

        For scalar-valued function, this is an empty tuple.
        """
        return self._output_shape

    @property
    def output_ndim(self) -> int:
        """Syntactic sugar for ``len(output_shape)``."""
        return self._output_ndim

    def __call__(self, x: ArrayLike) -> np.ndarray:
        """Evaluate the function at a given input.

        The function is vectorized over the batch shape of the input.

        Parameters
        ----------
        x
            *shape=* ``batch_shape +`` :attr:`input_shape` -- (Batch of) input(s) at
            which to evaluate the function.

        Returns
        -------
        fx
            *shape=* ``batch_shape +`` :attr:`output_shape` -- Function evaluated at
            the given (batch of) input(s).

        Raises
        ------
        ValueError
            If the shape of ``x`` does not match :attr:`input_shape` along its last
            dimensions.
        """
        x = np.asarray(x)

        # Shape checking
        if x.shape[x.ndim - self.input_ndim :] != self.input_shape:
            err_msg = (
                "The shape of the input array must match the `input_shape` "
                f"`{self.input_shape}` of the function along its last dimensions, but "
                f"an array with shape `{x.shape}` was given."
            )

            raise ValueError(err_msg)

        batch_shape = x.shape[: x.ndim - self.input_ndim]

        fx = self._evaluate(x)

        assert fx.shape == (batch_shape + self.output_shape)

        return fx

    @abc.abstractmethod
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        pass

    def __neg__(self):
        return -1.0 * self

    @functools.singledispatchmethod
    def __add__(self, other):
        return NotImplemented

    @functools.singledispatchmethod
    def __sub__(self, other):
        return NotImplemented

    @functools.singledispatchmethod
    def __mul__(self, other):
        if np.ndim(other) == 0:
            # pylint: disable=import-outside-toplevel,cyclic-import
            from ._algebra_fallbacks import ScaledFunction

            return ScaledFunction(function=self, scalar=other)

        return NotImplemented

    @functools.singledispatchmethod
    def __rmul__(self, other):
        if np.ndim(other) == 0:
            # pylint: disable=import-outside-toplevel,cyclic-import
            from ._algebra_fallbacks import ScaledFunction

            return ScaledFunction(function=self, scalar=other)

        return NotImplemented


class LambdaFunction(Function):
    """Define a :class:`Function` from a given :class:`callable`.

    Creates a :class:`Function` from a given :class:`callable` and in- and output
    shapes. This provides a convenient interface to define a :class:`Function`.

    Parameters
    ----------
    fn
        Callable defining the function.
    input_shape
        Input shape.
    output_shape
        Output shape.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.functions import LambdaFunction
    >>> fn = LambdaFunction(fn=lambda x: 2 * x + 1, input_shape=(2,), output_shape=(2,))
    >>> fn(np.array([[1, 2], [4, 5]]))
    array([[ 3,  5],
           [ 9, 11]])

    See Also
    --------
    Function : Callable with a fixed in- and output shape.
    """

    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        input_shape: ShapeLike,
        output_shape: ShapeLike = (),
    ) -> None:
        self._fn = fn

        super().__init__(input_shape, output_shape)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._fn(x)
