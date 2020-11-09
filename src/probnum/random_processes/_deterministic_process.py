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
    input_shape :
        Shape of an input to the deterministic process.
    output_shape :
        Shape of the output of the deterministic process.
    dtype :
        Data type of the random process evaluated at an input. If ``object`` will be
        converted to ``numpy.dtype``.
    fun :
        Callable defining the deterministic process.
    random_state :
        Random state of the random process. If None (or np.random), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.

    See Also
    --------
    RandomProcess : Class representing random processes.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum import random_processes as rps
    >>> f = lambda x : 2.0 * x ** 2 - 1.25 * x + 5.0
    >>> rp = rps.DeterministicProcess(fun=f,
    ...                               input_shape=(),
    ...                               output_shape=(),
    ...                               dtype=np.dtype(np.float_))
    >>> x = np.linspace(0, 1, 10)
    >>> np.all(rp.sample(x) == f(x))
    True
    """

    def __init__(
        self,
        input_shape: ShapeArgType,
        output_shape: ShapeArgType,
        dtype: DTypeArgType,
        fun: Callable[[_InputType], _OutputType],
        random_state: RandomStateArgType = None,
    ):
        self._fun = fun

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=dtype,
            random_state=random_state,
            fun=fun,
            sample_at_input=self._sample_at_input,
            mean=fun,
            cov=self._cov,
            var=self._var,
        )

    def _cov(self, x0: _InputType, x1: _InputType, keepdims=False) -> _OutputType:

        if keepdims:
            cov_shape = (
                (x0.shape[0], x1.shape[0]) + self.output_shape + self.output_shape
            )
        else:
            cov_shape = (
                x0.shape[0] * np.prod(self.output_shape),
                x1.shape[0] * np.prod(self.output_shape),
            )

        return np.zeros(shape=cov_shape)

    def _var(self, x: _InputType) -> _OutputType:
        return np.zeros(shape=[x.shape[0], self.output_shape])

    def _sample_at_input(self, x: _InputType, size: ShapeArgType = ()) -> _OutputType:
        return np.tile(self.__call__(x), reps=size)
