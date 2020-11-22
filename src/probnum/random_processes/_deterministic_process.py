"""Random process wrapper class for (deterministic) functions."""

from typing import Callable, TypeVar

import numpy as np

import probnum.random_variables as rvs
from probnum.type import DTypeArgType, IntArgType, RandomStateArgType, ShapeArgType

from . import _random_process

_InputType = TypeVar("InputType")
_OutputType = TypeVar("ValueType")


class DeterministicProcess(_random_process.RandomProcess[_InputType, _OutputType]):
    """Random process representing a deterministic function.

    Deterministic function wrapped as a
    :class:`~probnum.random_processes.RandomProcess`.

    This class has the useful property that arithmetic operations between a
    :class:`DeterministicProcess` and an arbitrary
    :class:`~probnum.random_processes.RandomProcess` represent the same arithmetic
    operation with an arbitrary function.

    Parameters
    ----------
    fun :
        Callable defining the deterministic process.
    input_dim :
        Shape of an input to the deterministic process.
    output_dim :
        Shape of the output of the deterministic process.
    dtype :
        Data type of the random process evaluated at an input. If ``object`` will be
        converted to ``numpy.dtype``.
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
    >>> rp = rps.DeterministicProcess(fun=f)
    >>> x = np.linspace(0, 1, 10)
    >>> np.all(rp.sample(x) == f(x))
    True
    """

    def __init__(
        self,
        fun: Callable[[_InputType], _OutputType],
        input_dim: IntArgType = 1,
        output_dim: IntArgType = 1,
        dtype: DTypeArgType = np.float_,
        random_state: RandomStateArgType = None,
    ):
        self._fun = fun

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=dtype,
            random_state=random_state,
            sample_at_input=self._sample_at_input,
            mean=self.__call__,
        )

    def __call__(self, x: _InputType) -> rvs.Constant[_OutputType]:
        return rvs.Constant(self.mean(x))

    def mean(self, x: _InputType) -> _OutputType:
        x = np.asarray(x)
        if x.ndim > 1:
            fun_eval = self._fun(x).reshape(x.shape[0], self.output_dim)
        else:
            fun_eval = self._fun(x).reshape(self.output_dim)
        return fun_eval

    def cov(self, x0: _InputType, x1: _InputType = None) -> _OutputType:
        x0 = np.atleast_2d(x0)
        if x1 is None:
            x0 = x1
        else:
            x1 = np.atleast_2d(x1)

        if self.output_dim == 1:
            cov_shape = (x0.shape[0], x1.shape[1])
        else:
            cov_shape = (x0.shape[0], x1.shape[1], self.output_dim, self.output_dim)

        return np.zeros(shape=cov_shape)

    def var(self, x: _InputType) -> _OutputType:
        x = np.asarray(x)
        if x.ndim == 1:
            var_shape = self.output_dim
        else:
            var_shape = (x.shape[0], self.output_dim)
        return np.zeros(shape=var_shape)

    def _sample_at_input(self, x: _InputType, size: ShapeArgType = ()) -> _OutputType:
        return np.tile(self.__call__(x), reps=size)
