"""ODE solver as proposed by Abdulle and Garegnani."""
import numpy as np
from pn_ode_benchmarks import noisy_step_rules, scipy_solution
from scipy.integrate._ivp import rk

from probnum import diffeq, randvars
from probnum.diffeq.perturbedsolvers import perturbedstepsolution


class PerturbedStepSolver(diffeq.ODESolver):
    """ODE Solver based on Scipy that introduces uncertainty by perturbing the time-
    steps."""

    # pylint: disable=maybe-no-member
    def __init__(self, solver, noise_scale, perturb_function, random_state=None):
        def perturbation_step(step):
            return perturb_function(
                step=step,
                solver_order=solver.order,
                noise_scale=noise_scale,
                random_state=random_state,
                size=(),
            )

        self.perturb_step = perturbation_step
        self.solver = solver
        self.scipy_solver = solver.solver
        self.interpolants = None
        self.evaluated_times = None
        self.posjected_times = None
        self.original_t = None
        self.scale = None
        self.scales = None
        super().__init__(ivp=solver.ivp, order=solver.order)

    def initialise(self):
        """Initialise and reset the solver."""
        self.interpolants = []
        self.evaluated_times = [self.solver.ivp.t0]
        self.projected_times = [self.solver.ivp.t0]
        self.scales = []
        self.scale = 0
        self.original_t = 0
        return self.solver.initialise()

    def step(self, start, stop, current, **kwargs):
        """Perform one perturbed ODE-step from start to perturbed stop.

        Parameters
        ----------
        start : float
            starting timepoint of the step
        stop : float
            stopping timepoint of the step
        current : :obj:`list` of :obj:`RandomVariable`
            current states of the ODE.

        Returns
        -------
        random_var : :obj:`list` of :obj:`RandomVariable`
            estimated states (in the state-space model view) of the discrete-time solution..
        error_estimation : float
            estimated error after having performed the step.
        """
        self.original_t = stop
        stepsize = stop - start
        noisy_step = self.perturb_step(stepsize)
        y_new, f_new = rk.rk_step(
            self.scipy_solver.fun,
            start,
            current.mean,
            self.scipy_solver.f,
            noisy_step,
            self.scipy_solver.A,
            self.scipy_solver.B,
            self.scipy_solver.C,
            self.scipy_solver.K,
        )
        error_estimation = self.scipy_solver._estimate_error(
            self.scipy_solver.K, noisy_step
        )
        random_var = randvars.Constant(y_new)
        self.scipy_solver.h_previous = stepsize
        self.scipy_solver.y_old = current
        # those values are used to compute the dense output of the original solution
        # which is rescaled in noisy_step_solution
        self.scipy_solver.t_old = start
        self.scipy_solver.t = start + noisy_step
        self.scipy_solver.y = y_new
        self.scipy_solver.h_abs = stepsize
        self.scipy_solver.f = f_new
        print(noisy_step)
        self.scale = noisy_step / stepsize
        return random_var, error_estimation

    def method_callback(self, time, current_guess, current_error):
        """calculates dense output after each step and stores it, stores the perturned
        timepoints at which the solution was evaluated and the oiginal timepoints to
        which the perturbed solution is projected."""
        self.projected_times.append(self.original_t)
        self.evaluated_times.append(self.scipy_solver.t)
        self.scales.append(self.scale)
        return self.solver.method_callback(time, current_guess, current_error)

    def rvlist_to_odesol(self, times, rvs):
        interpolants = self.solver.interpolants
        # those are the timepoints at which the solution was actually evaluated
        projected_times = self.projected_times
        # those are the timepoints on which we project the solution that we actually evaluated at evaluated_timepoints
        evaluated_times = self.evaluated_times
        # scales
        scales = self.scales
        probnum_solution = perturbedstepsolution.PerturbedStepSolution(
            scales, projected_times, rvs, interpolants
        )
        return probnum_solution

    def postprocess(self, odesol):
        return odesol
