"""ODE solutions returned by Gaussian ODE filtering."""

import typing

import numpy as np

import probnum._randomvariablelist as pnrv_list
import probnum.filtsmooth as pnfs
import probnum.type
import probnum.utils
from probnum import randvars
from probnum.utils.linalg import cholesky_update

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
    >>> from probnum import randvars
    >>>
    >>> def f(t, x):
    ...     return 4*x*(1-x)
    >>>
    >>> y0 = np.array([0.15])
    >>> t0, tmax = 0., 1.5
    >>> solution = probsolve_ivp(f, t0, tmax, y0, step=0.1, adaptive=False)
    >>> # Mean of the discrete-time solution
    >>> print(np.round(solution.y.mean, 2))
    [[0.15]
     [0.21]
     [0.28]
     [0.37]
     [0.47]
     [0.57]
     [0.66]
     [0.74]
     [0.81]
     [0.87]
     [0.91]
     [0.94]
     [0.96]
     [0.97]
     [0.98]
     [0.99]]

    >>> # Times of the discrete-time solution
    >>> print(solution.t)
    [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.  1.1 1.2 1.3 1.4 1.5]
    >>> # Individual entries of the discrete-time solution can be accessed with
    >>> print(solution[5])
    <Normal with shape=(1,), dtype=float64>
    >>> print(np.round(solution[5].mean, 2))
    [0.56]
    >>> # Evaluate the continuous-time solution at a new time point t=0.65
    >>> print(np.round(solution(0.65).mean, 2))
    [0.70]
    """

    def __init__(self, kalman_posterior: pnfs.KalmanPosterior):
        self.kalman_posterior = kalman_posterior

        # Pre-compute projection matrices.
        # The prior must be an integrator, if not, an error is thrown in 'GaussianIVPFilter'.
        self.proj_to_y = kalman_posterior.transition.proj2coord(coord=0)
        self.proj_to_dy = kalman_posterior.transition.proj2coord(coord=1)

    @property
    def t(self) -> np.ndarray:
        return self.kalman_posterior.locations

    @cached_property
    def y(self) -> pnrv_list._RandomVariableList:
        y_rvs = [
            _project_rv(self.proj_to_y, rv) for rv in self.kalman_posterior.state_rvs
        ]
        return pnrv_list._RandomVariableList(y_rvs)

    @cached_property
    def dy(self) -> pnrv_list._RandomVariableList:
        dy_rvs = [
            _project_rv(self.proj_to_dy, rv) for rv in self.kalman_posterior.state_rvs
        ]
        return pnrv_list._RandomVariableList(dy_rvs)

    def __call__(
        self, t: float
    ) -> typing.Union[randvars.RandomVariable, pnrv_list._RandomVariableList]:
        out_rv = self.kalman_posterior(t)

        if np.isscalar(t):
            return _project_rv(self.proj_to_y, out_rv)

        return pnrv_list._RandomVariableList(
            [_project_rv(self.proj_to_y, rv) for rv in out_rv]
        )

    def sample(
        self,
        t: typing.Optional[float] = None,
        size: typing.Optional[probnum.type.ShapeArgType] = (),
    ) -> np.ndarray:
        """Sample from the Gaussian filtering ODE solution by sampling from the Gauss-
        Markov posterior."""
        size = probnum.utils.as_shape(size)

        # implement only single samples, rest via recursion
        # We cannot 'steal' the recursion from self.kalman_posterior.sample,
        # because we need to project the respective states out of each sample.
        if size != ():
            return np.array([self.sample(t=t, size=size[1:]) for _ in range(size[0])])

        samples = self.kalman_posterior.sample(locations=t, size=size)
        return np.array([self.proj_to_y @ sample for sample in samples])

    @property
    def filtering_solution(self):

        if isinstance(self.kalman_posterior, pnfs.FilteringPosterior):
            return self

        # else: self.kalman_posterior is a SmoothingPosterior object, which has the field filter_posterior.
        return KalmanODESolution(
            kalman_posterior=self.kalman_posterior.filtering_posterior
        )


def _project_rv(projmat, rv):
    # There is no way of checking whether `rv` has its Cholesky factor computed already or not.
    # Therefore, since we need to update the Cholesky factor for square-root filtering,
    # we also update the Cholesky factor for non-square-root algorithms here,
    # which implies additional cost.
    # See Issues #319 and #329.
    # When they are resolved, this function here will hopefully be superfluous.

    new_mean = projmat @ rv.mean
    new_cov = projmat @ rv.cov @ projmat.T
    new_cov_cholesky = cholesky_update(projmat @ rv.cov_cholesky)
    return randvars.Normal(new_mean, new_cov, cov_cholesky=new_cov_cholesky)
