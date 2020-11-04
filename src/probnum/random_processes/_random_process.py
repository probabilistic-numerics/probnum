"""
Random Processes.
"""

from typing import Callable, Generic, Optional, TypeVar

import numpy as np

from probnum import utils as _utils
from probnum.type import (
    DTypeArgType,
    RandomStateArgType,
    RandomStateType,
    ShapeArgType,
    ShapeType,
)

from ..random_variables import RandomVariable

_DomainType = TypeVar("DomainType")
_ValueType = TypeVar("ValueType")


class RandomProcess(Generic[_DomainType, _ValueType]):
    """
    Random processes represent uncertainty about a function.

    Random processes generalize functions by encoding uncertainty over function
    values in their covariance function. Random processes can be used to model
    (deterministic) functions which are not fully known or describe functions with
    stochastic output.

    Instances of :class:`RandomProcess` can be added, multiplied, etc. with scalars,
    arrays and random variables. Such (arithmetic) operations may not retain all
    previously available methods.

    Parameters
    ----------
    value_shape :
        Shape of the random process evaluated at an input.
    dtype :
        Data type of the random process evaluated at an input. If ``object`` will be
        converted to ``numpy.dtype``.
    random_state :
        Random state of the random process. If None (or np.random), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.

    See Also
    --------
    asrandproc : Convert an object to a random process.

    Notes
    -----
    Sampling from random processes with fixed seed is not stable with respect to the
    order of operations. This means sampling from a random process and then
    performing an arithmetic operation will not necessarily return the same samples
    as if the order of operations is reversed.
    """

    def __init__(
        self,
        input_shape: ShapeArgType,
        value_shape: ShapeArgType,
        dtype: DTypeArgType,
        random_state: RandomStateArgType = None,
        call: Optional[Callable[[_DomainType], RandomVariable[_ValueType]]] = None,
        sample_at_input: Optional[
            Callable[[_DomainType, ShapeType], _ValueType]
        ] = None,
        mean: Optional[Callable[[_DomainType], _ValueType]] = None,
        cov: Optional[Callable[[_DomainType], _ValueType]] = None,
        var: Optional[Callable[[_DomainType], _ValueType]] = None,
        std: Optional[Callable[[_DomainType], _ValueType]] = None,
    ):
        """Create a new random process."""
        # Shape and data type
        self.__input_shape = _utils.as_shape(input_shape)
        self.__value_shape = _utils.as_shape(value_shape)
        self.__dtype = np.dtype(dtype)

        # Random seed and sampling
        self._random_state = _utils.as_random_state(random_state)
        self.__sample_at_input = sample_at_input

        # Functions of the random process
        self.__mean = mean
        self.__cov = cov
        self.__var = var
        self.__std = std

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with value shape={self.value_shape}, dtype"
            f"={self.dtype}>"
        )

    def __call__(self, input) -> RandomVariable:
        """
        Evaluate the random process at a set of inputs.

        Parameters
        ----------
        input
        """
        raise NotImplementedError

    @property
    def value_shape(self) -> ShapeType:
        """Shape of the random process evaluated at an input."""
        return self.__value_shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of (elements of) the random process evaluated at an input."""
        return self.__dtype

    @property
    def random_state(self) -> RandomStateType:
        """Random state of the random process.

        This attribute defines the RandomState object to use for drawing
        realizations from this random variable.
        If None (or np.random), the global np.random state is used.
        If integer, it is used to seed the local :class:`~numpy.random.RandomState`
        instance.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed: RandomStateArgType):
        """
        Get or set the RandomState object of the random process.

        This can be either None or an existing RandomState object.
        If None (or np.random), use the RandomState singleton used by np.random.
        If already a RandomState instance, use it.
        If an int, use a new RandomState instance seeded with seed.
        """
        self._random_state = _utils.as_random_state(seed)

    def mean(self, input: _DomainType) -> _ValueType:
        raise NotImplementedError

    def std(self, input: _DomainType) -> _ValueType:
        raise NotImplementedError

    def var(self, input: _DomainType) -> _ValueType:
        raise NotImplementedError

    def cov(self, input1: _DomainType, input2: _DomainType) -> _ValueType:
        raise NotImplementedError

    def sample(self, input: _DomainType = None, size: ShapeArgType = ()):
        """
        Sample paths from the random process.

        Parameters
        ----------
        input
            Evaluation input of the sample paths of the process.
        size
            Size of the sample.
        """
        if input is None:
            return lambda inp: self.sample(input=inp, size=size)
        else:
            return self._sample_at_input(input=input, size=size)

    def _sample_at_input(self, input: _DomainType, size: ShapeArgType = ()):
        raise NotImplementedError
