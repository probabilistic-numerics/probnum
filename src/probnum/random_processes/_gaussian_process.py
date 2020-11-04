"""Gaussian processes."""

from typing import Callable, Optional, Union

import numpy as np

from probnum import utils as _utils
from probnum.filtsmooth.statespace import DiscreteGaussianLinearModel, LinearSDEModel
from probnum.type import ShapeArgType

from ..random_variables import Normal, RandomVariable
from . import _random_process

_DomainType = Union[np.floating, np.ndarray]
_ValueType = Union[np.floating, np.ndarray]


class GaussianProcess(_random_process.RandomProcess[_DomainType, _ValueType]):
    """
    Gaussian processes.

    A Gaussian process is a continuous stochastic process which if evaluated at a
    finite set of inputs returns a multivariate normal random variable. Gaussian
    processes are fully characterized by their mean and covariance function.

    Parameters
    ----------

    See Also
    --------
    RandomProcess : Class representing random processes.

    Examples
    --------
    >>> import numpy as np
    >>> mean = lambda x : np.zeros_like(x)
    >>> kernel = lambda x, y : (x @ y) ** 2
    >>> gp = GaussianProcess(mean=mean, kernel=kernel, input_shape=())
    >>> gp.sample(input=np.linspace(0, 1, 5))
    <Normal with shape=(5,), dtype=float64>
    """

    def __init__(
        self,
        mean: Optional[Callable[[_DomainType], _ValueType]],
        cov: Optional[Callable[[_DomainType], _ValueType]],
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
            value_shape=_value_shape,
            dtype=np.dtype(np.float_),
            mean=mean,
            cov=cov,
        )

    def __call__(self, input: _DomainType) -> Normal:
        return Normal(mean=self.mean(input), cov=self.cov(input1=input, input2=input))

    def mean(self, input: _DomainType) -> _ValueType:
        return self.mean(input)

    def std(self, input: _DomainType) -> _ValueType:
        raise NotImplementedError

    def var(self, input: _DomainType) -> _ValueType:
        raise NotImplementedError

    def cov(self, input1: _DomainType, input2: _DomainType) -> _ValueType:
        return self.cov(input1, input2)

    def _sample_at_input(self, input: _DomainType, size: ShapeArgType = ()):
        return Normal(
            mean=self.mean(input), cov=self.cov(input1=input, input2=input)
        ).sample(size=size)


class GaussMarkovProcess(_random_process.RandomProcess[np.floating, _ValueType]):
    """
    Gauss-Markov Processes.


    Parameters
    ----------

    See Also
    --------
    RandomProcess : Class representing random processes.

    Examples
    --------
    """

    def __init__(self, linear_transition, initrv, t0=0.0):
        if not isinstance(linear_transition, LinearSDEModel):
            raise ValueError
        elif not isinstance(linear_transition, DiscreteGaussianLinearModel):
            raise ValueError
        self.transition = linear_transition
        self.t0 = t0
        self.initrv = initrv
        super().__init__(mean=self._sde_meanfun, cov=self._covfun)

    def _sde_meanfun(self, input):
        return self._transition_rv(input).mean

    def _sde_covfun(self, input1, input2):
        raise NotImplementedError

    def _transition_rv(self, input):
        return self.transition.transition_rv(rv=self.initrv, start=self.t0, stop=input)

    @property
    def varfun(self):
        return lambda loc: self._transition_rv(loc).cov
