"""
Random Processes.
"""

from typing import Callable, Generic, Optional, TypeVar, Union

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

_InputType = TypeVar("InputType")
_OutputType = TypeVar("ValueType")


class RandomProcess(Generic[_InputType, _OutputType]):
    """
    Random processes represent uncertainty about a function.

    Random processes generalize functions by encoding uncertainty over function
    values in their covariance function. They can be used to model (deterministic)
    functions which are not fully known or to define functions with stochastic output.

    Instances of :class:`RandomProcess` can be added, multiplied, etc. with scalars,
    arrays and random variables. Such (arithmetic) operations may not retain all
    previously available methods.

    Parameters
    ----------
    input_shape :
        Shape of an input to the random process.
    output_shape :
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
    as if the order of operations is reversed. However, the random seed ensures that
    each sequence of operations will always result in the same output.
    """

    def __init__(
        self,
        input_shape: ShapeArgType,
        output_shape: ShapeArgType,
        dtype: DTypeArgType,
        random_state: RandomStateArgType = None,
        call: Optional[Callable[[_InputType], RandomVariable[_OutputType]]] = None,
        sample_at_input: Optional[
            Callable[[_InputType, ShapeType], _OutputType]
        ] = None,
        mean: Optional[Callable[[_InputType], _OutputType]] = None,
        cov: Optional[Callable[[_InputType, _InputType], _OutputType]] = None,
        var: Optional[Callable[[_InputType], _OutputType]] = None,
        std: Optional[Callable[[_InputType], _OutputType]] = None,
    ):
        """Create a new random process."""
        # Evaluation of the random process
        self.__call = call

        # Shape and data type
        self.__input_shape = _utils.as_shape(input_shape)
        self.__output_shape = _utils.as_shape(output_shape)
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
            f"<{self.__class__.__name__} with value shape={self.output_shape}, dtype"
            f"={self.dtype}>"
        )

    def __call__(self, input: _InputType) -> RandomVariable[_OutputType]:
        """
        Evaluate the random process at a set of inputs.

        Parameters
        ----------
        input
            Inputs to evaluate random process at.
        """
        self.__call(input)

    @property
    def input_shape(self) -> ShapeType:
        """Shape of inputs to the random process."""
        return self.__output_shape

    @property
    def output_shape(self) -> ShapeType:
        """Shape of the random process evaluated at an input."""
        return self.__output_shape

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

    def mean(self, input: _InputType) -> _OutputType:
        """
        Mean function of the random process.

        Parameters
        ----------
        input
            Inputs where the mean function is evaluated.
        """
        if self.__mean is None:
            raise NotImplementedError

        return self.__mean(input)

    def std(self, input: _InputType) -> _OutputType:
        """
        Standard deviation function of the random process.

        Parameters
        ----------
        input
            Input locations.
        """
        if self.__std is None:
            try:
                return np.sqrt(self.var(input=input))
            except NotImplementedError as exc:
                raise NotImplementedError from exc
        else:
            return self.__std(input)

    def var(self, input: _InputType) -> _OutputType:
        """
        Variance function of the random process.

        Returns the value of the covariance or kernel evaluated pairwise at `input`.

        Parameters
        ----------
        input
            Input locations.
        """
        if self.__var is None:
            try:
                return (
                    np.diag(self.cov(input0=input, input1=input))
                    .reshape(input.shape)
                    .copy()
                )
            except NotImplementedError as exc:
                raise NotImplementedError from exc
        else:
            return self.__var(input)

    def cov(self, input0: _InputType, input1: _InputType) -> _OutputType:
        """
        Covariance function or kernel of the random process.

        Parameters
        ----------
        input0
            First input to the covariance function.
        input1
            Second input to the covariance function.
        """
        if self.__cov is None:
            raise NotImplementedError

        return self.__cov(input0, input1)

    def sample(
        self, input: _InputType = None, size: ShapeArgType = ()
    ) -> Union[Callable[[_InputType], _OutputType], RandomVariable[_OutputType]]:
        """
        Sample paths from the random process.

        If no inputs are provided this function returns sample paths which are
        callables, otherwise random variables corresponding to the input locations
        are returned.

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

    def _sample_at_input(
        self, input: _InputType, size: ShapeArgType = ()
    ) -> RandomVariable[_OutputType]:
        """
        Evaluate a set of sample paths at the given inputs.

        This function should be implemented by subclasses of :class:`RandomProcess`.
        This enables :meth:`sample` to both return functions, i.e. sample paths if
        only a `size` is provided and random variables if inputs are provided as well.

        Parameters
        ----------
        input
            Evaluation input of the sample paths of the process.
        size
            Size of the sample.
        """
        if self.__sample_at_input is None:
            raise NotImplementedError("No sampling method provided.")

        return self.__sample_at_input(input, _utils.as_shape(size))
