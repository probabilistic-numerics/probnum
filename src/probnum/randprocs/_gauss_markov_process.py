"""Gauss-Markov Processes."""
from typing import Union

import numpy as np

from probnum import randvars
from probnum.type import FloatArgType, RandomStateArgType

from ._gaussian_process import GaussianProcess

_InputType = Union[np.floating, np.ndarray]
_OutputType = Union[np.floating, np.ndarray]


class GaussMarkovProcess(GaussianProcess):
    """Gaussian processes with the Markov property.

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
    t0
        Initial starting index / time of the process.
    initrv
        Gaussian random variable describing the initial state.

    See Also
    --------
    RandomProcess : Random processes.
    GaussianProcess : Gaussian processes.

    Examples
    --------
    """

    def __init__(
        self,
        linear_transition: Union,
        initrv: randvars.Normal,
        t0: FloatArgType = 0.0,
    ):
        self.transition = linear_transition
        self.t0 = t0
        self.initrv = initrv
        super().__init__(
            input_dim=1,
            output_dim=initrv.shape[0],
            mean=self._sde_solution_mean,
            cov=self._sde_solution_cov,
        )

    # TODO: currently the SDE Models do not support arbitrary steps as defined by
    #  the increments in ``x``, but rather only predefined "euler_step"s. This has to
    #  be added first before __call__, mean, etc. work as defined by the
    #  RandomProcess interface.

    def __call__(self, x: _InputType) -> randvars.Normal:
        """Closed form solution to the SDE evaluated at ``x`` as defined by the linear
        transition."""
        return self.transition.transition_rv(rv=self.initrv, start=self.t0, stop=x)

    def _sde_solution_mean(self, x: _InputType) -> _OutputType:
        return self.__call__(x).mean

    def _sde_solution_cov(self, x: _InputType) -> _OutputType:
        return self.__call__(x).cov
