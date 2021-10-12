"""Wrapper class of scipy.integrate. for RK23 and RK45.

Dense-output can not be used for DOP853, if you use other RK-methods, make sure, that
the current implementation works for them.
"""
import numpy as np
from scipy.integrate._ivp import rk
from scipy.integrate._ivp.common import OdeSolution

from probnum import randvars
from probnum.diffeq import _odesolver, _odesolver_state
from probnum.diffeq.perturbed.scipy_wrapper import _wrapped_scipy_odesolution
from probnum.typing import FloatArgType


class WrappedScipyRungeKutta(_odesolver.ODESolver):
    """Wrapper for Runge-Kutta methods from SciPy."""

    def __init__(self, solver_type: rk.RungeKutta, steprule):
        self.solver_type = solver_type
        self.interpolants = None

        # Filled in later
        self.solver = None
        self.ivp = None

        # Dopri853 as implemented in SciPy computes the dense output differently.
        if issubclass(solver_type, rk.DOP853):
            raise TypeError(
                "Dense output interpolation of DOP853 is currently not supported. Choose a different RK-method."
            )

        super().__init__(steprule=steprule, order=solver_type.order)

    def initialize(self, ivp):
        """Return t0 and y0 (for the solver, which might be different to ivp.y0) and
        initialize the solver. Reset the solver when solving the ODE multiple times,
        i.e. explicitly setting y_old, t, y and f to the respective initial values,
        otherwise those are wrong when running the solver twice.

        Returns
        -------
        self.ivp.t0: float
            initial time point
        self.ivp.initrv: randvars.RandomVariable
            initial random variable
        """
        self.solver = self.solver_type(ivp.f, ivp.t0, ivp.y0, ivp.tmax)
        self.ivp = ivp

        self.interpolants = []
        self.solver.y_old = None
        self.solver.t = self.ivp.t0
        self.solver.y = self.ivp.y0
        self.solver.f = self.solver.fun(self.solver.t, self.solver.y)
        state = _odesolver_state.ODESolverState(
            ivp=ivp,
            rv=randvars.Constant(self.ivp.y0),
            t=self.ivp.t0,
            error_estimate=None,
            reference_state=None,
        )
        return state

    def attempt_step(self, state: _odesolver_state.ODESolverState, dt: FloatArgType):
        """Perform one ODE-step from start to stop and set variables to the
        corresponding values.

        To specify start and stop directly, rk_step() and not _step_impl() is used.

        Parameters
        ----------
        state
            Current state of the ODE solver.
        dt
            Step-size.

        Returns
        -------
        _odesolver_state.ODESolverState
            New state.
        """

        y_new, f_new = rk.rk_step(
            self.solver.fun,
            state.t,
            state.rv.mean,
            self.solver.f,
            dt,
            self.solver.A,
            self.solver.B,
            self.solver.C,
            self.solver.K,
        )

        # Unnormalized error estimation is used as the error estimation is normalized in
        # solve().
        error_estimation = self.solver._estimate_error(self.solver.K, dt)
        y_new_as_rv = randvars.Constant(y_new)
        new_state = _odesolver_state.ODESolverState(
            ivp=state.ivp,
            rv=y_new_as_rv,
            t=state.t + dt,
            error_estimate=error_estimation,
            reference_state=state.rv.mean,
        )

        # Update the solver settings. This part is copied from scipy's _step_impl().
        self.solver.h_previous = dt
        self.solver.y_old = state.rv.mean
        self.solver.t_old = state.t
        self.solver.t = state.t + dt
        self.solver.y = y_new
        self.solver.h_abs = dt
        self.solver.f = f_new

        return new_state

    def rvlist_to_odesol(self, times: np.array, rvs: np.array):
        """Create a ScipyODESolution object which is a subclass of
        diffeq.ODESolution."""
        scipy_solution = OdeSolution(times, self.interpolants)
        probnum_solution = _wrapped_scipy_odesolution.WrappedScipyODESolution(
            scipy_solution, rvs
        )
        return probnum_solution

    def method_callback(self, state):
        """Call dense output after each step and store the interpolants."""
        dense = self.dense_output()
        self.interpolants.append(dense)

    def dense_output(self):
        """Compute the interpolant after each step.

        Returns
        -------
        sol : rk.RkDenseOutput
            Interpolant between the last and current location.
        """
        Q = self.solver.K.T.dot(self.solver.P)
        sol = rk.RkDenseOutput(self.solver.t_old, self.solver.t, self.solver.y_old, Q)
        return sol
