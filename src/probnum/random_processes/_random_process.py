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
_OutputType = TypeVar("OutputType")


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
        Shape :math:`d` of an input to the random process.
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

    # pylint: disable=too-many-instance-attributes

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
        # pylint: disable=too-many-arguments
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

    def __call__(self, x: _InputType) -> RandomVariable[_OutputType]:
        """
        Evaluate the random process at a set of inputs.

        Parameters
        ----------
        x
            *shape=(d,) or (n, d)* -- Input(s) to evaluate random process at.
        """
        if self.__call is None:
            raise NotImplementedError

        return self.__call(x)

    @property
    def input_shape(self) -> ShapeType:
        """Shape of inputs to the random process."""
        return self.__input_shape

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

    def mean(self, x: _InputType) -> _OutputType:
        """
        Mean function of the random process.

        Parameters
        ----------
        x
            *shape=(d,) or (n, d)* -- Inputs where the mean function is evaluated.
        """
        if self.__mean is None:
            raise NotImplementedError

        return self.__mean(x)

    def std(self, x: _InputType) -> _OutputType:
        """
        Standard deviation function of the random process.

        Parameters
        ----------
        x
            *shape=(d,) or (n, d)* -- Input locations.

        Returns
        -------
        std
            *shape=(n,) or (n, d)* -- Standard deviation of the process at ``x``.
        """
        if self.__std is None:
            try:
                return np.sqrt(self.var(x=x))
            except NotImplementedError as exc:
                raise NotImplementedError from exc
        else:
            return self.__std(x)

    def var(self, x: _InputType) -> _OutputType:
        """
        Variance function of the random process.

        Returns the variance function which is the value of the covariance or kernel
        evaluated pairwise at ``x`` for each output dimension separately.

        Parameters
        ----------
        x
            *shape=(d,) or (n, d)* -- Input locations.

        Returns
        -------
        var
            *shape=(n,) or (n, d)* -- Variance of the process at ``x``.
        """
        if self.__var is None:
            try:
                return np.diag(self.cov(x0=x, x1=x)).reshape(x.shape).copy()
            except NotImplementedError as exc:
                raise NotImplementedError from exc
        else:
            return self.__var(x)

    def cov(self, x0: _InputType, x1: _InputType) -> _OutputType:
        """
        Covariance function or kernel of the random process.

        Returns the covariance function :math:`\\operatorname{Cov}(x_0,
        x_1) = \\mathbb{E}[(f(x_0) - \\mathbb{E}[f(x_0)])(f(x_0) - \\mathbb{E}[f(
        x_0)])^\\top]` of the process evaluated at ``x0`` and ``x1``. The resulting
        covariance has *shape=(n0, n1) or (n0, n1, d, d)* in the case of
        multi-dimensional output.

        Parameters
        ----------
        x0
            *shape=(d,) or (n0, d)* -- First input to the covariance function.
        x1
            *shape=(d,) or (n1, d)* -- Second input to the covariance function.

        Returns
        -------
        cov
            *shape=(n0, n1) or (n0, n1, d, d)* -- Covariance of the process at ``x0``
            and ``x1``.
        """  # pylint: disable=trailing-whitespace
        if self.__cov is None:
            raise NotImplementedError

        return self.__cov(x0, x1)

    def sample(
        self, x: _InputType = None, size: ShapeArgType = ()
    ) -> Union[Callable[[_InputType], _OutputType], RandomVariable[_OutputType]]:
        """
        Sample paths from the random process.

        If no inputs are provided this function returns sample paths which are
        callables, otherwise random variables corresponding to the input locations
        are returned.

        Parameters
        ----------
        x
            *shape=(d,) or (n, d)* -- Evaluation input(s) of the sample paths of the
            process.
        size
            Size of the sample.
        """
        if x is None:
            return lambda x: self.sample(x=x, size=size)

        return self._sample_at_input(x=x, size=size)

    def _sample_at_input(
        self, x: _InputType, size: ShapeArgType = ()
    ) -> RandomVariable[_OutputType]:
        """
        Evaluate a set of sample paths at the given inputs.

        This function should be implemented by subclasses of :class:`RandomProcess`.
        This enables :meth:`sample` to both return functions, i.e. sample paths if
        only a `size` is provided and random variables if inputs are provided as well.

        Parameters
        ----------
        x
            *shape=(d,) or (n, d)* -- Evaluation input(s) of the sample paths of the
            process.
        size
            Size of the sample.
        """
        if self.__sample_at_input is None:
            raise NotImplementedError("No sampling method provided.")

        return self.__sample_at_input(x, _utils.as_shape(size))
