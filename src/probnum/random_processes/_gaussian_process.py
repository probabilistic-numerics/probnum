"""Gaussian processes."""

from functools import lru_cache
from typing import Callable

from probnum.filtsmooth.statespace import DiscreteGaussianLinearModel, LinearSDEModel
from probnum.type import ShapeArgType

from ..random_variables import Normal, RandomVariable
from ._random_process import RandomProcess


class GaussianProcess(RandomProcess):
    def __init__(self, meanfun, covfun):
        self._meanfun = meanfun
        self._covfun = covfun

    def __call__(self, location) -> RandomVariable:
        """
        Evaluate the random process at a set of inputs.

        Parameters
        ----------
        location

        Returns
        -------

        """
        return Normal(mean=self.meanfun(location), cov=self.covfun(location))

    @property
    def mean(self) -> Callable:
        return self._meanfun

    @property
    def std(self) -> Callable:
        raise NotImplementedError

    def var(self) -> Callable:  # varfun(pt)
        raise NotImplementedError

    @property
    def cov_function(self) -> Callable:  # covfun(pt1, pt2)
        return self._covfun

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


class GaussMarkovProcess(GaussianProcess):
    def __init__(self, linear_transition, initrv, t0=0.0):
        if not isinstance(linear_transition, LinearSDEModel):
            raise ValueError
        elif not isinstance(linear_transition, DiscreteGaussianLinearModel):
            raise ValueError
        self.transition = linear_transition
        self.t0 = t0
        self.initrv = initrv
        super().__init__(meanfun=self._sde_meanfun, covfun=self._covfun)

    def _sde_meanfun(self, location):
        return self._transition_rv(location).mean

    def _sde_covfun(self, location1, location2):
        raise NotImplementedError

    @lru_cache
    def _transition_rv(self, location):
        return self.transition.transition_rv(
            rv=self.initrv, start=self.t0, stop=location
        )

    @property
    def varfun(self):
        return lambda loc: self._transition_rv(loc).cov
