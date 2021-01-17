"""ODE solutions returned by Gaussian ODE filtering."""

import typing

import numpy as np

import probnum._randomvariablelist as pnrv_list
import probnum.filtsmooth as pnfs
import probnum.random_variables as pnrv
import probnum.type
import probnum.utils

from ..odesolution import ODESolution

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


class KalmanODESolution(ODESolution):
    """Gaussian IVP filtering solution of an ODE problem.

    Recall that in ProbNum, Gaussian filtering and smoothing is generally named "Kalman".

    Parameters
    ----------
    kalman_posterior
        Gauss-Markov posterior over the ODE solver state space model.
        Therefore, it assumes that the dynamics model is an :class:`Integrator`.

    See Also
    --------
    GaussianIVPFilter : ODE solver that behaves like a Gaussian filter.
    KalmanPosterior : Posterior over states after Gaussian filtering/smoothing.

    Examples
    --------
    >>> from probnum.diffeq import logistic, probsolve_ivp
    >>> from probnum import random_variables as rvs
    >>> initrv = rvs.Constant(0.15)
    >>> ivp = logistic(timespan=[0., 1.5], initrv=initrv, params=(4, 1))
    >>> solution = probsolve_ivp(ivp, method="ekf0", step=0.1)
    >>> # Mean of the discrete-time solution
    >>> print(solution.y.mean)
    [[0.15      ]
     [0.2076198 ]
     [0.27932997]
     [0.3649165 ]
     [0.46054129]
     [0.55945475]
     [0.65374523]
     [0.73686744]
     [0.8053776 ]
     [0.85895587]
     [0.89928283]
     [0.92882899]
     [0.95007559]
     [0.96515825]
     [0.97577054]
     [0.9831919 ]]
    >>> # Times of the discrete-time solution
    >>> print(solution.t)
    [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.  1.1 1.2 1.3 1.4 1.5]
    >>> # Individual entries of the discrete-time solution can be accessed with
    >>> print(solution[5])
    <Normal with shape=(1,), dtype=float64>
    >>> print(solution[5].mean)
    [0.55945475]
    >>> # Evaluate the continuous-time solution at a new time point t=0.65
    >>> print(solution(0.65).mean)
    [0.69875089]
    """

    def __init__(self, kalman_posterior: pnfs.KalmanPosterior):
        self.kalman_posterior = kalman_posterior

        # Pre-compute projection matrices.
        # The prior must be an integrator, if not, an error is thrown in 'GaussianIVPFilter'.
        self.proj_to_y = kalman_posterior.gauss_filter.dynamics_model.proj2coord(
            coord=0
        )
        self.proj_to_dy = kalman_posterior.gauss_filter.dynamics_model.proj2coord(
            coord=1
        )

    def append(self, t, y):
        raise NotImplementedError

    @property
    def t(self) -> np.ndarray:
        return self.kalman_posterior.locations

    @cached_property
    def y(self) -> pnrv_list._RandomVariableList:
        y_rvs = [self.proj_to_y @ rv for rv in self.kalman_posterior.state_rvs]
        return pnrv_list._RandomVariableList(y_rvs)

    @cached_property
    def dy(self) -> pnrv_list._RandomVariableList:
        dy_rvs = [self.proj_to_dy @ rv for rv in self.kalman_posterior.state_rvs]
        return pnrv_list._RandomVariableList(dy_rvs)

    def __call__(
        self, t: float
    ) -> typing.Union[pnrv.RandomVariable, pnrv_list._RandomVariableList]:
        out_rv = self.kalman_posterior(t)

        if np.isscalar(t):
            return self.proj_to_y @ out_rv

        return pnrv_list._RandomVariableList([self.proj_to_y @ rv for rv in out_rv])

    def sample(
        self,
        t: typing.Optional[float] = None,
        size: typing.Optional[probnum.type.ShapeArgType] = (),
    ) -> np.ndarray:
        """Sample from the Gaussian filtering ODE solution by sampling from the Gauss-
        Markov posterior."""
        size = probnum.utils.as_shape(size)

        # implement only single samples, rest via recursion
        if size != ():
            return np.array([self.sample(t=t, size=size[1:]) for _ in range(size[0])])

        samples = self.kalman_posterior.sample(locations=t, size=size)
        return np.array([self.proj_to_y @ sample for sample in samples])
