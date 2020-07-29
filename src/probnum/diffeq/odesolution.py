"""ODESolution object, returned by `probsolve_ivp`

Contains the discrete time and function outputs.
Provides dense output by being callable.
Can function values can also be accessed by indexing.
"""
import numpy as np

from probnum.prob import RandomVariable, Normal
from probnum.prob.randomvariablelist import _RandomVariableList
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior
from probnum.filtsmooth import KalmanPosterior


class ODESolution(FiltSmoothPosterior):
    """Solution of an ODE problem

    Examples
    --------
    >>> from probnum.diffeq import logistic, probsolve_ivp
    >>> from probnum.prob import RandomVariable, Dirac, Normal
    >>> initrv = RandomVariable(distribution=Dirac(0.15))
    >>> ivp = logistic(timespan=[0., 1.5], initrv=initrv, params=(4, 1))
    >>> solution = probsolve_ivp(ivp, method="ekf0", step=0.1)
    >>> # Mean of the discrete-time solution
    >>> print(solution.y.mean())
    [[0.15       0.51      ]
     [0.2076198  0.642396  ]
     [0.27932997 0.79180747]
     [0.3649165  0.91992313]
     [0.46054129 0.9925726 ]
     [0.55945475 0.98569653]
     [0.65374523 0.90011316]
     [0.73686744 0.76233098]
     [0.8053776  0.60787222]
     [0.85895587 0.4636933 ]
     [0.89928283 0.34284592]
     [0.92882899 0.24807715]
     [0.95007559 0.17685497]
     [0.96515825 0.12479825]
     [0.97577054 0.08744746]
     [0.9831919  0.06097975]]
    >>> # Times of the discrete-time solution
    >>> print(solution.t)
    [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.  1.1 1.2 1.3 1.4 1.5]
    >>> # Individual entries of the discrete-time solution can be accessed with
    >>> print(solution[5])
    <(2,) RandomVariable with dtype=<class 'float'>>
    >>> print(solution[5].mean())
    [0.55945475 0.98569653]
    >>> # Evaluate the continuous-time solution at a new time point t=0.65
    >>> print(solution(0.65).mean())
    [0.69702861 0.83122207]
    """

    def __init__(self, times, rvs, solver):
        self._state_posterior = KalmanPosterior(times, rvs, solver.gfilt)
        self._solver = solver

    def _proj_normal_rv(self, rv, coord):
        """Projection of a normal RV, e.g. to map 'states' to 'function values'"""
        q = self._solver.prior.ordint
        new_mean = rv.mean()[coord :: (q + 1)]
        new_cov = rv.cov()[coord :: (q + 1), coord :: (q + 1)]
        return RandomVariable(distribution=Normal(new_mean, new_cov))

    @property
    def t(self):
        """Times of the discrete-time solution"""
        return self._state_posterior.locations

    @property
    def y(self):
        """Probabilistic discrete-time solution, as a list of random variables

        To return means and covariances use `y.mean()` and `y.cov()`.
        """
        function_rvs = [self._proj_normal_rv(rv, 0) for rv in self._state_rvs]
        return _RandomVariableList(function_rvs)

    @property
    def dy(self):
        """Derivatives of the discrete-time solution, as a list of random variables"""
        dy_rvs = [self._proj_normal_rv(rv, 1) for rv in self._state_rvs]
        return _RandomVariableList(dy_rvs)

    @property
    def _state_rvs(self):
        """Time-discrete posterior estimates over states, without preconditioning"""
        state_rvs = _RandomVariableList(
            [self._solver.undo_preconditioning_rv(rv) for rv in self._state_posterior]
        )
        return state_rvs

    def __call__(self, t, smoothed=True):
        """
        Evaluate the time-continuous solution at time t

        `KalmanPosterior.__call__` does the main algorithmic work to return the
        posterior for a given location. All that is left to do here is to (1) undo the
        preconditioning, and (2) to slice the state_rv in order to return only the
        rv for the function value.
        """
        out_rv = self._state_posterior(t, smoothed=smoothed)
        out_rv = self._solver.undo_preconditioning_rv(out_rv)
        out_rv = self._proj_normal_rv(out_rv, 0)
        return out_rv

    def __len__(self):
        """Number of points in the discrete-time solution"""
        return len(self._state_posterior)

    def __getitem__(self, idx):
        """Access the discrete-time solution through indexing and slicing"""
        if isinstance(idx, int):
            rv = self._state_posterior[idx]
            rv = self._solver.undo_preconditioning_rv(rv)
            rv = self._proj_normal_rv(rv, 0)
            return RandomVariable(distribution=Normal(f_mean, f_cov))
        elif isinstance(idx, slice):
            rvs = self._state_posterior[idx]
            rvs = [self._solver.undo_preconditioning_rv(rv) for rv in rvs]
            f_rvs = [self._proj_normal_rv(rv, 0) for rv in rvs]
            return _RandomVariableList(f_rvs)
        else:
            raise ValueError("Invalid index")
