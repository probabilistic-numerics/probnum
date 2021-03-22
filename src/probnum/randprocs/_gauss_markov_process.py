"""Gauss-Markov Processes."""
from typing import Union

import numpy as np

from probnum import randvars, statespace
from probnum.type import FloatArgType

from ._gaussian_process import GaussianProcess

_InputType = Union[np.floating, np.ndarray]
_OutputType = Union[np.floating, np.ndarray]


class GaussMarkovProcess(GaussianProcess):
    r"""Gaussian processes with the Markov property.

    A Gauss-Markov process is a Gaussian process with the additional property that
    conditioned on the present state of the system its future and past states are
    independent. This is known as the Markov property or as the process being
    memoryless. Gauss-Markov processes :math:`x_t` can be defined as the solution of a
    (linear) transition model given by a stochastic differential equation (SDE) of
    the form

    .. math:: \mathrm{d} x_t = G(t) x_t \mathrm{d} t + \mathrm{d} w_t.

    and a Gaussian initial condition :math:`x_0`.

    Parameters
    ----------
    linear_sde
        Linear stochastic differential equation describing a state transition of the
        system.
    t0
        Initial starting index / time of the process.
    x0
        Gaussian random variable describing the initial state.

    See Also
    --------
    RandomProcess : Random processes.
    GaussianProcess : Gaussian processes.

    Examples
    --------
    """
    # pylint: disable=invalid-name

    def __init__(
        self,
        linear_sde: statespace.LinearSDE,
        x0: randvars.Normal,
        t0: FloatArgType = 0.0,
    ):
        self.linear_sde = linear_sde
        self.t0 = t0
        self.x0 = np.asarray(x0).reshape(1, -1)

        super().__init__(
            input_dim=1,
            output_dim=1 if self.x0.ndim == 0 else self.x0.shape[0],
            mean=lambda t: self.__call__(t).mean,
            cov=lambda t: self.__call__(t).cov,
        )

    def __call__(self, x: _InputType) -> randvars.Normal:
        """Closed form solution to the SDE evaluated at ``x`` as defined by the linear
        transition."""
        # TODO: currently the SDE Models do not support arbitrary steps as defined by
        #  the increments in ``x``, only fixed step sizes ``dt```. This has to
        #  be added first before __call__, mean, etc. work as defined by the
        #  RandomProcess interface.

        # return self.linear_sde.forward_rv(rv=self.x0, t=self.t0, dt=x[-1] - x[-2])
        raise NotImplementedError
