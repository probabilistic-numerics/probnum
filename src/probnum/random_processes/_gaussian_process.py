"""Gaussian processes."""

from typing import Callable, Optional, Union

import numpy as np

from probnum import utils as _utils
from probnum.filtsmooth.statespace import DiscreteGaussianLinearModel, LinearSDEModel
from probnum.type import FloatArgType, ShapeArgType

from ..random_variables import Normal, RandomVariable
from . import _random_process

_InputType = Union[np.floating, np.ndarray]
_OutputType = Union[np.floating, np.ndarray]


class GaussianProcess(_random_process.RandomProcess[_InputType, _OutputType]):
    """
    Gaussian processes.

    A Gaussian process is a continuous stochastic process which if evaluated at a
    finite set of inputs returns a multivariate normal random variable. Gaussian
    processes are fully characterized by their mean and covariance function.

    Parameters
    ----------
    mean :
        Mean function.
    cov :
        Covariance function or kernel.
    input_shape :
        Shape of the input of the Gaussian process.

    See Also
    --------
    RandomProcess : Class representing random processes.
    GaussMarkovProcess : Gaussian processes with the Markov property.

    Examples
    --------
    >>> import numpy as np
    >>> mean = lambda x : np.zeros_like(x)  # zero-mean function
    >>> kernel = lambda x, y : (x @ y) ** 2  # polynomial kernel
    >>> gp = GaussianProcess(mean=mean, cov=kernel, input_shape=())
    >>> gp.sample(input=np.linspace(0, 1, 5))
    <Normal with shape=(5,), dtype=float64>
    """

    def __init__(
        self,
        mean: Optional[Callable[[_InputType], _OutputType]],
        cov: Optional[Callable[[_InputType], _OutputType]],
        input_shape: ShapeArgType,
    ):
        # Type normalization
        # TODO

        # Data type normalization
        # TODO

        # Shape checking
        _input_shape = _utils.as_shape(input_shape)
        _value_shape = mean(np.zeros_like(input_shape)).shape

        # Call to super class
        super().__init__(
            input_shape=_input_shape,
            output_shape=_value_shape,
            dtype=np.dtype(np.float_),
            mean=mean,
            cov=cov,
        )

    def __call__(self, input: _InputType) -> Normal:
        return Normal(mean=self.mean(input), cov=self.cov(input0=input, input1=input))

    def _sample_at_input(self, input: _InputType, size: ShapeArgType = ()):
        return Normal(
            mean=self.mean(input), cov=self.cov(input0=input, input1=input)
        ).sample(size=size)


class GaussMarkovProcess(GaussianProcess):
    """
    Gaussian processes with the Markov property.

    A Gauss-Markov process is a Gaussian process with the additional property that
    conditioned on the present state of the system its future and past states are
    independent. This is known as the Markov property or as the process being
    memoryless.

    Parameters
    ----------
    linear_transition
        Linear transition model describing a state change of the system.
    initrv
        Initial random variable describing the initial state.
    t0
        Initial starting index / time of the process.

    See Also
    --------
    GaussianProcess : Class representing Gaussian processes.

    Examples
    --------
    """

    def __init__(
        self,
        linear_transition: Union[LinearSDEModel, DiscreteGaussianLinearModel],
        initrv: RandomVariable[_OutputType],
        t0: FloatArgType = 0.0,
    ):
        self.transition = linear_transition
        self.t0 = t0
        self.initrv = initrv
        super().__init__(input_shape=(), mean=self._sde_meanfun, cov=self._sde_covfun)

    def _sde_meanfun(self, input):
        return self._transition_rv(input).mean

    def _sde_covfun(self, input0, input1):
        raise NotImplementedError

    def _transition_rv(self, input):
        return self.transition.transition_rv(rv=self.initrv, start=self.t0, stop=input)

    def var(self, input):
        return lambda loc: self._transition_rv(input).cov
