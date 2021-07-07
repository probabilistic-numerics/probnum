"""Wrapper class of scipy.integrate. for RK23 and RK45.

Dense-output can not be used for DOP853, if you use other RK-methods,
make sure, that the current implementation works for them.
"""
import numpy as np
from scipy.integrate._ivp import rk
from scipy.integrate._ivp.common import OdeSolution

from probnum import diffeq, randvars
from probnum.diffeq import wrappedscipyodesolution
from probnum.typing import FloatArgType


class WrappedScipyRungeKutta(diffeq.ODESolver):
    """Wrapper for Runge-Kutta methods from Scipy, implements the stepfunction and dense
    output."""

    def __init__(self, solver: rk.RungeKutta):
        self.solver = solver
        self.interpolants = None

        # ProbNum ODESolver needs an ivp
        ivp = diffeq.IVP(
            timespan=[self.solver.t, self.solver.t_bound],
            initrv=randvars.Constant(self.solver.y),
            rhs=self.solver._fun,
        )

        # Dopri853 as implemented in SciPy computes the dense output differently.
        if isinstance(solver, rk.DOP853):
            raise TypeError(
                "Dense output interpolation of DOP853 is currently not supported. Choose a different RK-method."
            )
        super().__init__(ivp=ivp, order=solver.order)

    def initialise(self):
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

        self.interpolants = []
        self.solver.y_old = None
        self.solver.t = self.ivp.t0
        self.solver.y = self.ivp.initrv.mean
        self.solver.f = self.solver.fun(self.solver.t, self.solver.y)
        return self.ivp.t0, self.ivp.initrv

    def step(
        self, start: FloatArgType, stop: FloatArgType, current: randvars, **kwargs
    ):
        """Perform one ODE-step from start to stop and set variables to the
        corresponding values.

        To specify start and stop directly, rk_step() and not _step_impl() is used.

        Parameters
        ----------
        start : float
            starting location of the step
        stop : float
            stopping location of the step
        current : :obj:`list` of :obj:`RandomVariable`
            current state of the ODE.

        Returns
        -------
        random_var : randvars.RandomVariable
            Estimated states of the discrete-time solution.
        error_estimation : float
            estimated error after having performed the step.
        """

        y = current.mean
        dt = stop - start
        y_new, f_new = rk.rk_step(
            self.solver.fun,
            start,
            y,
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

        # Update the solver settings. This part is copied from scipy's _step_impl().
        self.solver.h_previous = dt
        self.solver.y_old = current
        self.solver.t_old = start
        self.solver.t = stop
        self.solver.y = y_new
        self.solver.h_abs = dt
        self.solver.f = f_new
        return y_new_as_rv, error_estimation, y

    def rvlist_to_odesol(self, times: np.array, rvs: np.array):
        """Create a ScipyODESolution object which is a subclass of
        diffeq.ODESolution."""
        scipy_solution = OdeSolution(times, self.interpolants)
        probnum_solution = wrappedscipyodesolution.WrappedScipyODESolution(
            scipy_solution, rvs
        )
        return probnum_solution

    def method_callback(self, time, current_guess, current_error):
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
        sol = rk.RkDenseOutput(
            self.solver.t_old, self.solver.t, self.solver.y_old.mean, Q
        )
        return sol
