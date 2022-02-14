"""Function class defining a Callable with fixed in- and output shapes."""

import abc
from typing import Callable

import numpy as np

from . import utils
from .typing import ArrayLike, ShapeLike, ShapeType


class Function(abc.ABC):
    """Callable with information about the shape of expected in- and outputs.

    This class represents a, uni- or multivariate, scalar- or tensor-valued,
    mathematical function. Hence, the call method should not have any observable
    side-effects.

    Parameters
    ----------
    input_shape
        Shape to which inputs are broadcast.

    output_shape
        Output shape.

    See Also
    --------
    LambdaFunction : Define a :class:`Function` from an anonymous function.
    ~probnum.randprocs.mean_fns.Zero : Zero mean function of a random process.
    """

    def __init__(self, input_shape: ShapeLike, output_shape: ShapeLike = ()) -> None:
        self._input_shape = utils.as_shape(input_shape)
        self._input_ndim = len(self._input_shape)

        self._output_shape = utils.as_shape(output_shape)
        self._output_ndim = len(self._output_shape)

    @property
    def input_shape(self) -> ShapeType:
        """Shape to which inputs are broadcast.

        For a scalar-input function, this is an empty tuple."""
        return self._input_shape

    @property
    def output_shape(self) -> ShapeType:
        """Shape of the function's output.

        For scalar-valued function, this is an empty tuple."""
        return self._output_shape

    def __call__(self, x: ArrayLike) -> np.ndarray:
        """Evaluate the function at a given input.

        Parameters
        ----------
        x
            *shape=*``batch_shape + input_shape_comp`` -- (Batch of) input(s) at which
            to evaluate the function. ``input_shape_comp`` must be a shape that can be
            broadcast to :attr:`input_shape`.

        Returns
        -------
        fx
            *shape=*``batch_shape + output_shape`` -- Function evaluated at the (batch
            of) input(s).
        """
        x = np.asarray(x)

        try:
            np.broadcast_to(
                x,
                shape=x.shape[: x.ndim - self._input_ndim] + self._input_shape,
            )
        except ValueError as ve:
            raise ValueError(
                f"The shape of the input {x.shape} is not compatible with the "
                f"specified `input_shape` of the `Function` {self._input_shape}."
            ) from ve

        res = self._evaluate(x)

        assert res.shape == (x.shape[: x.ndim - self._input_ndim] + self._output_shape)

        return res

    @abc.abstractmethod
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        pass


class LambdaFunction(Function):
    """Define a :class:`Function` from an anonymous function.

    Creates a :class:`Function` from an anonymous function and in- and output shapes.
    This provides a convenient interface to define a :class:`Function`.

    Parameters
    ----------
    fn
        Callable defining the function.
    input_shape
        Shape to which inputs are broadcast.
    output_shape
        Output shape.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum import LambdaFunction
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
        output_shape: ShapeLike,
    ) -> None:
        self._fn = fn

        super().__init__(input_shape, output_shape)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._fn(x)
