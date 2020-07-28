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
    """Continuous ODE Solution"""

    def __init__(self, times, rvs, solver):
        self._state_posterior = KalmanPosterior(times, rvs, solver.gfilt)
        self._solver = solver
        self._d = self._solver.ivp.ndim

    @property
    def t(self):
        """Times of the discrete-time solution"""
        return self._state_posterior.locations

    @property
    def y(self):
        """Probabilistic discrete-time solution, as a list of random variables"""
        function_rvs = [
            RandomVariable(
                distribution=Normal(
                    rv.mean()[0 :: self._d], rv.cov()[0 :: self._d, 0 :: self._d]
                )
            )
            for rv in self._state_rvs
        ]
        return _RandomVariableList(function_rvs)

    @property
    def dy(self):
        """Derivatives of the discrete-time solution, as a list of random variables"""
        function_rvs = [
            RandomVariable(
                distribution=Normal(
                    rv.mean()[1 :: self._d], rv.cov()[1 :: self._d, 1 :: self._d]
                )
            )
            for rv in self._state_rvs
        ]
        return _RandomVariableList(function_rvs)

    @property
    def _state_rvs(self):
        """Time-discrete posterior estimates over states, without preconditioning"""
        state_rvs = _RandomVariableList(
            [self._solver.undo_preconditioning_rv(rv) for rv in self._state_posterior]
        )
        return state_rvs

    def __call__(self, t, smoothed=True):
        """
        Evaluate the solution at time t

        `KalmanPosterior.__call__` does the main algorithmic work to return the
        posterior for a given location. All that is left to do here is to (1) undo the
        preconditioning, and (2) to slice the state_rv in order to return only the
        rv for the function value.
        """

        out_rv = self._state_posterior(t, smoothed=smoothed)
        out_rv = self._solver.undo_preconditioning_rv(out_rv)

        f_mean = out_rv.mean()[0 :: self._d]
        f_cov = out_rv.cov()[0 :: self._d, 0 :: self._d]

        return RandomVariable(distribution=Normal(f_mean, f_cov))

    def __len__(self):
        """Number of points in the discrete-time solution"""
        return len(self._state_posterior)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            rv = self._state_posterior[idx]
            rv = self._solver.undo_preconditioning_rv(rv)
            f_mean = rv.mean()[0 :: self._d]
            f_cov = rv.cov()[0 :: self._d, 0 :: self._d]
            return RandomVariable(distribution=Normal(f_mean, f_cov))
        elif isinstance(idx, slice):
            rvs = self._state_posterior[idx]
            rvs = [self._solver.undo_preconditioning_rv(rv) for rv in rvs]
            f_means = [rv.mean()[0 :: self._d] for rv in rvs]
            f_covs = [rv.cov()[0 :: self._d, 0 :: self._d] for rv in rvs]
            f_rvs = [
                RandomVariable(distribution=Normal(f_mean, f_cov))
                for (f_mean, f_cov) in zip(f_means, f_covs)
            ]
            return _RandomVariableList(f_rvs)
        else:
            raise ValueError("Invalid index")
