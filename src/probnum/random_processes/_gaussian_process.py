"""Gaussian processes."""

from functools import lru_cache
from typing import Callable, Union

import numpy as np

from probnum.filtsmooth.statespace import DiscreteGaussianLinearModel, LinearSDEModel
from probnum.type import ShapeArgType

from ..random_variables import Normal, RandomVariable
from . import _random_process

_DomainType = Union[np.floating, np.ndarray]
_ValueType = Union[np.floating, np.ndarray]


class GaussianProcess(_random_process.RandomProcess[_DomainType, _ValueType]):
    def __init__(self, meanfun, covfun):
        self._meanfun = meanfun
        self._covfun = covfun

    def __call__(self, input) -> RandomVariable:
        """
        Evaluate the random process at a set of inputs.

        Parameters
        ----------
        input

        Returns
        -------

        """
        return Normal(mean=self._meanfun(input), cov=self._covfun(input))

    def mean(self, input: _DomainType) -> _ValueType:
        return self._meanfun(input)

    def std(self, input: _DomainType) -> _ValueType:
        raise NotImplementedError

    def var(self, input: _DomainType) -> _ValueType:
        raise NotImplementedError

    def cov(self, input1: _DomainType, input2: _DomainType) -> _ValueType:
        return self._covfun(input1, input2)

    def sample_function(self, size: ShapeArgType = ()) -> Callable:
        """
        Sample an instance from the random process.

        Parameters
        ----------
        size
            Size of the sample.
        """
        # return lambda loc: self.__call__(loc).sample(size=size)
        raise NotImplementedError


class GaussMarkovProcess(_random_process.RandomProcess[np.floating, _ValueType]):
    """"""

    def __init__(self, linear_transition, initrv, t0=0.0):
        if not isinstance(linear_transition, LinearSDEModel):
            raise ValueError
        elif not isinstance(linear_transition, DiscreteGaussianLinearModel):
            raise ValueError
        self.transition = linear_transition
        self.t0 = t0
        self.initrv = initrv
        super().__init__(meanfun=self._sde_meanfun, covfun=self._covfun)

    def _sde_meanfun(self, input):
        return self._transition_rv(input).mean

    def _sde_covfun(self, input1, input2):
        raise NotImplementedError

    @lru_cache
    def _transition_rv(self, input):
        return self.transition.transition_rv(rv=self.initrv, start=self.t0, stop=input)

    @property
    def varfun(self):
        return lambda loc: self._transition_rv(loc).cov
