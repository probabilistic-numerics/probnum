"""ODESolution object, returned by `probsolve_ivp`

Contains the discrete time and function outputs. Provides dense output
by being callable. Can function values can also be accessed by indexing.
"""
import numpy as np

from probnum import utils
from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth import KalmanPosterior
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior


class ODESolution(FiltSmoothPosterior):
    """Gaussian IVP filtering solution of an ODE problem.

    Parameters
    ----------
    times : `array_like`
        Times of the discrete-time solution.
    rvs : :obj:`list` of :obj:`RandomVariable`
        Estimated states (in the state-space model view) of the discrete-time solution.
    solver : :obj:`GaussianIVPFilter`
        Solver used to compute the discrete-time solution.


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

    def __init__(self, times, rvs, solver):

        # try-except is a hotfix for now:
        # future PR is to move KalmanPosterior-info out of here, e.g. into GaussianIVPFilter
        try:
            self._kalman_posterior = KalmanPosterior(
                times, rvs, solver.gfilt, solver.with_smoothing
            )
            self._t = None
            self._y = None
        except AttributeError:
            self._kalman_posterior = None
            self._t = times
            self._y = _RandomVariableList(rvs)
        self._solver = solver

    @property
    def t(self):
        """:obj:`np.ndarray`: Times of the discrete-time solution"""
        if self._t:  # hotfix
            return self._t
        else:
            return self._kalman_posterior.locations

    @property
    def y(self):
        """:obj:`list` of :obj:`RandomVariable`: Probabilistic discrete-time solution

        Probabilistic discrete-time solution at times :math:`t_1, ..., t_N`,
        as a list of random variables.
        To return means and covariances use ``y.mean`` and ``y.cov``.
        """
        if self._y:  # hotfix
            return self._y
        else:
            projmat = self._solver.prior.proj2coord(coord=0)
            function_rvs = [projmat @ rv for rv in self._state_rvs]
            return _RandomVariableList(function_rvs)

    @property
    def dy(self):
        """:obj:`list` of :obj:`RandomVariable`: Derivatives of the discrete-time solution"""
        projmat = self._solver.prior.proj2coord(coord=1)
        dy_rvs = [projmat @ rv for rv in self._state_rvs]
        return _RandomVariableList(dy_rvs)

    @property
    def _state_rvs(self):
        """:obj:`list` of :obj:`RandomVariable`:"""
        return self._kalman_posterior.state_rvs

    def __call__(self, t):
        """Evaluate the time-continuous solution at time t.

        `KalmanPosterior.__call__` does the main algorithmic work to return the
        posterior for a given location. All that is left to do here is to (1) undo the
        preconditioning, and (2) to slice the state_rv in order to return only the
        rv for the function value.

        Parameters
        ----------
        t : float
            Location / time at which to evaluate the continuous ODE solution.

        Returns
        -------
        :obj:`RandomVariable`
            Probabilistic estimate of the continuous-time solution at time ``t``.
        """
        out_rv = self._kalman_posterior(t)
        projmat = self._solver.prior.proj2coord(coord=0)

        if np.isscalar(t):
            return projmat @ out_rv

        return _RandomVariableList([projmat @ rv for rv in out_rv])

    def __len__(self):
        """Number of points in the discrete-time solution."""
        return len(self._kalman_posterior)

    def __getitem__(self, idx):
        """Access the discrete-time solution through indexing and slicing."""
        projmat = self._solver.prior.proj2coord(coord=0)

        if isinstance(idx, int):
            rv = self._kalman_posterior[idx]
            return projmat @ rv
        elif isinstance(idx, slice):
            rvs = self._kalman_posterior[idx]
            f_rvs = [projmat @ rv for rv in rvs]
            return _RandomVariableList(f_rvs)
        else:
            raise ValueError("Invalid index")

    def sample(self, t=None, size=()):

        size = utils.as_shape(size)
        projmat = self._solver.prior.proj2coord(coord=0)

        # implement only single samples, rest via recursion
        if size != ():
            return np.array([self.sample(t=t, size=size[1:]) for _ in range(size[0])])

        samples = self._kalman_posterior.sample(locations=t, size=size)
        return np.array([projmat @ sample for sample in samples])
