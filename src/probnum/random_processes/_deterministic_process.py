"""
Random process wrapper class for (deterministic) functions.
"""

from typing import Callable, TypeVar

import numpy as np

from probnum.type import DTypeArgType, RandomStateArgType, ShapeArgType

from . import _random_process

_InputType = TypeVar("InputType")
_OutputType = TypeVar("ValueType")


class DeterministicProcess(_random_process.RandomProcess[_InputType, _OutputType]):
    """
    Random process representing a deterministic function.

    Deterministic function wrapped as a
    :class:`~probnum.random_processes.RandomProcess`.

    This class has the useful property that arithmetic operations between a
    :class:`DeterministicProcess` and an arbitrary
    :class:`~probnum.random_processes.RandomProcess` represent the same arithmetic
    operation with an arbitrary function.

    Parameters
    ----------

    See Also
    --------
    RandomProcess : Class representing random processes.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum import random_processes as rps
    >>> f = lambda x : 2.0 * x ** 2 - 1.25 * x + 5.0
    >>> rp = rps.DeterministicProcess(f, input_shape=(), output_shape=(), dtype=float)
    >>> x = np.linspace(0, 1, 10)
    >>> np.all(rp.sample(x) == f(x))
    True
    """

    def __init__(
        self,
        fun: Callable[[_InputType], _OutputType],
        input_shape: ShapeArgType,
        output_shape: ShapeArgType,
        dtype: DTypeArgType,
        random_state: RandomStateArgType = None,
    ):
        self._fun = fun

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=dtype,
            random_state=random_state,
            call=fun,
            sample_at_input=self.__sample_at_input,
            mean=fun,
            cov=lambda x: np.zeros_like(x, shape=(output_shape, output_shape)),
            var=lambda x: np.zeros_like(x),
        )

    def __sample_at_input(self, x: _InputType, size: ShapeArgType = ()) -> _OutputType:
        return np.tile(self.__call__(x), reps=size)
