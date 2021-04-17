"""Wrapper class of scipy.integrate."""
import scipy.integrate._ivp.rk as scipy_rk
from scipy.integrate._ivp.common import OdeSolution

from probnum import diffeq, randvars
from probnum.diffeq import scipyodesolution as scisol


class ScipyODESolver(diffeq.ODESolver):
    """Interface for scipy based ODE solvers."""

    def __init__(self, solver, order):
        self.solver = solver
        ivp = diffeq.IVP(
            timespan=[self.solver.t, self.solver.t_bound],
            initrv=randvars.Constant(self.solver.y),
            rhs=self.solver._fun,
        )
        self.function = self.solver._fun
        self.interpolants = None
        self.scipy_solution = None
        self.method = self.solver.__class__.__name__
        super().__init__(ivp=ivp, order=order)

    def initialise(self):
        """Returns t0 and y0 (for the solver, which might be different to ivp.y0) and
        initializes the solver. Resets when solving the ODE multiple times.

        Returns
        -------
        self.ivp.t0: float
            initial time point
        self.ivp.initrv: :obj:`list` of :obj:`RandomVariable`
            initial random variables
        """

        self.interpolants = []
        self.solver.y_old = None
        self.solver.t = self.ivp.t0
        self.solver.y = self.ivp.initrv.mean
        self.solver.f = self.solver.fun(self.solver.t, self.solver.y)
        return self.ivp.t0, self.ivp.initrv


class ScipyRungeKutta(ScipyODESolver):
    """Wraps Runge-Kutta methods from Scipy, implements the stepfunction and dense
    output."""

    def dense_output(self):
        """Returns the dense output after each step.

        Returns
        -------
        sol : scipy_rk.RkDenseOutput
            interpolated solution between two discrete locations.
        """

        if self.method == "RK45" or self.method == "RK23":
            sol = self.dense_output_rk()
        else:
            raise "Dense Output is not implemented for" + self.method
        return sol

    def dense_output_rk(self):
        """Computes the interpolant after each step with a quartic interpolation
        polynomial.

        Returns
        -------
        sol : scipy_rk.RkDenseOutput
            interpolated solution between two discrete locations.
        """

        Q = self.solver.K.T.dot(self.solver.P)
        sol = scipy_rk.RkDenseOutput(
            self.solver.t_old, self.solver.t, self.solver.y_old.mean, Q
        )
        return sol

    def step(self, start, stop, current, **kwargs):
        """Perform one ODE-step from start to stop.

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
        random_var : :obj:`list` of :obj:`RandomVariable`
            Estimated states (in the state-space model view) of the discrete-time solution..
        error_estimation : float
            estimated error after having performed the step.
        """
        y = current.mean
        # set stepsize
        h_abs = stop - start
        y_new, f_new = scipy_rk.rk_step(
            self.solver.fun,
            start,
            y,
            self.solver.f,
            h_abs,
            self.solver.A,
            self.solver.B,
            self.solver.C,
            self.solver.K,
        )
        error_estimation = self.solver._estimate_error(self.solver.K, h_abs)
        random_var = randvars.Constant(y_new)
        self.solver.h_previous = h_abs
        self.solver.y_old = current
        self.solver.t_old = start
        self.solver.t = stop
        self.solver.y = y_new
        self.solver.h_abs = h_abs
        self.solver.f = f_new
        return random_var, error_estimation

    def rvlist_to_odesol(self, times, rvs):
        """Create a ScipyODESolution object which is a subclass of
        diffeq.ODESolution."""
        self.solver.scipy_solution = OdeSolution(times, self.interpolants)
        probnum_solution = scisol.ScipyODESolution(
            self.solver.scipy_solution, times, rvs
        )
        return probnum_solution

    def method_callback(self, time, current_guess, current_error):
        """Call dense output after each step and store the interpolants."""
        dense = self.dense_output()
        self.interpolants.append(dense)
