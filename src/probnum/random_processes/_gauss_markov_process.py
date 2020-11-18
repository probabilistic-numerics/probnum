"""Gauss-Markov Processes."""
from typing import Union

import numpy as np

from probnum.filtsmooth.statespace import DiscreteLinearGaussian, LinearSDE
from probnum.random_variables import Normal
from probnum.type import FloatArgType, RandomStateArgType

from ._gaussian_process import GaussianProcess

_InputType = Union[np.floating, np.ndarray]
_OutputType = Union[np.floating, np.ndarray]


class GaussMarkovProcess(GaussianProcess):
    """
    Gaussian processes with the Markov property.

    A Gauss-Markov process is a Gaussian process with the additional property that
    conditioned on the present state of the system its future and past states are
    independent. This is known as the Markov property or as the process being
    memoryless. Gauss-Markov processes :math:`x_t` can be defined as the solution of a
    (linear) transition model given by a stochastic differential equation (SDE) of
    the form

    .. math:: d x_t = G(t) x_t d t + d w_t.

    and a Gaussian initial condition.

    Parameters
    ----------
    linear_transition
        Linear transition model describing a state change of the system.
    x0
        Initial starting index / time of the process.
    initrv
        Gaussian random variable describing the initial state.
    random_state :
        Random state of the random process. If None (or np.random), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.

    See Also
    --------
    GaussianProcess : Class representing Gaussian processes.

    Examples
    --------

    """

    def __init__(
        self,
        linear_transition: Union[LinearSDE, DiscreteLinearGaussian],
        initrv: Normal,
        x0: FloatArgType = 0.0,
        random_state: RandomStateArgType = None,
    ):
        self.transition = linear_transition
        self.x0 = x0
        self.initrv = initrv
        super().__init__(
            input_shape=(),
            output_shape=initrv.shape,
            mean=self._sde_solution_mean,
            cov=self._sde_solution_cov,
            random_state=random_state,
        )

    # TODO: currently the SDE Models do not support arbitrary steps as defined by
    #  the increments in ``x``, but rather only predefined "euler_step"s. This has to
    #  be added first before __call__, mean, etc. work as defined by the
    #  RandomProcess interface.

    def __call__(self, x: _InputType) -> Normal:
        """
        Closed form solution to the SDE evaluated at ``x`` as defined by the
        linear transition.
        """
        return self.transition.transition_rv(rv=self.initrv, start=self.x0, stop=x)

    def _sde_solution_mean(self, x: _InputType) -> _OutputType:
        return self.__call__(x).mean

    def _sde_solution_cov(self, x: _InputType) -> _OutputType:
        return self.__call__(x).cov
