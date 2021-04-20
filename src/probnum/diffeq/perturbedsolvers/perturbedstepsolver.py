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
    def __init__(self, solver, noise_scale, perturb_function):
        self.solver = solver
        self.scipy_solver = solver.solver
        self.noise_scale = noise_scale
        self.interpolants = None
        self.evaluated_times = None
        self.posjected_times = None
        self.original_t = 0
        # self.perturb_function = perturb_function
        self.steprule = perturb_function
        super().__init__(ivp=solver.ivp, order=solver.order)

    def initialise(self):
        """initialise the solver."""
        self.interpolants = []
        self.evaluated_times = [self.solver.ivp.t0]
        self.projected_times = [self.solver.ivp.t0]
        return self.solver.initialise()

    def step(self, start, stop, current, **kwargs):
        """perform one perturbed ODE-step from start to noisy stop.

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
            Estimated states (in the state-space model view) of the discrete-time solution..
        error_estimation : float
            estimated error after having performed the step.
        """
        y = current.mean
        # set stepsize
        self.original_t = stop
        laststep = stop - start
        noisy_step = self.perturb(laststep)
        y_new, f_new = rk.rk_step(
            self.scipy_solver.fun,
            start,
            y,
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
        self.scipy_solver.h_previous = laststep
        self.scipy_solver.y_old = current
        # those values are used to calculate the dense output of the original solution which is rescaled in noisy_step_solution
        self.scipy_solver.t_old = start
        self.scipy_solver.t = start + noisy_step
        self.scipy_solver.y = y_new
        self.scipy_solver.h_abs = laststep
        self.scipy_solver.f = f_new
        return random_var, error_estimation

    def method_callback(self, time, current_guess, current_error):
        """calculates dense output after each step and stores it, stores the perturned
        timepoints at which the solution was evaluated and the oiginal timepoints to
        which the perturbed solution is projected."""
        self.projected_times.append(self.original_t)
        self.evaluated_times.append(self.scipy_solver.t)
        return self.solver.method_callback(time, current_guess, current_error)

    def rvlist_to_odesol(self, times, rvs):
        interpolants = self.solver.interpolants
        # those are the timepoints at which the solution was actually evaluated
        projected_times = self.projected_times
        # those are the timepoints on which we project the solution that we actually evaluated at evaluated_timepoints
        evaluated_times = self.evaluated_times
        probnum_solution = perturbedstepsolution.PerturbedStepSolution(
            projected_times, evaluated_times, rvs, interpolants
        )
        return probnum_solution

    def postprocess(self, odesol):
        return odesol

    def perturb(self, step):
        """perturbs the performed step according to the chosen steprule (uniform or
        lognormal), scaled by the chosen noise-scale."""
        if self.steprule == "uniform":
            if step < 1:
                noisy_step = np.random.uniform(
                    step - self.noise_scale * step ** (self.order + 0.5),
                    step + self.noise_scale * step ** (self.order + 0.5),
                )
            else:
                print("Error: Stepsize too large (>=1), not possible")
        if self.steprule == "log":
            mean = np.log(step) - np.log(
                np.sqrt(1 + self.noise_scale * (step ** (2 * self.order)))
            )
            cov = np.log(1 + self.noise_scale * (step ** (2 * self.order)))
            noisy_step = np.exp(np.random.normal(mean, cov))
        return noisy_step
