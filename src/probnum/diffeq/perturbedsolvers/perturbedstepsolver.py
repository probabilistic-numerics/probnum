"""ODE solver as proposed by Abdulle and Garegnani."""
import numpy as np
from pn_ode_benchmarks import noisy_step_rules, scipy_solution
from scipy.integrate._ivp import base, rk

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
        self.time = None
        self.scale = None
        self.scales = None
        super().__init__(ivp=solver.ivp, order=solver.order)

    def initialise(self):
        """Initialise and reset the solver."""
        self.interpolants = []
        self.scales = []
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
        dt = stop - start
        noisy_step = self.perturb_step(dt)
        state_as_rv, error_estimation = self.solver.step(
            start, start + noisy_step, current
        )
        self.scale = noisy_step / dt
        self.time = stop
        return state_as_rv, error_estimation

    def method_callback(self, time, current_guess, current_error):
        """Computes dense output after each step and stores it, stores the perturned
        timepoints at which the solution was evaluated and the oiginal timepoints to
        which the perturbed solution is projected."""
        self.scales.append(self.scale)
        return self.solver.method_callback(time, current_guess, current_error)

    def rvlist_to_odesol(self, times, rvs):
        interpolants = self.solver.interpolants
        probnum_solution = perturbedstepsolution.PerturbedStepSolution(
            self.scales, times, rvs, interpolants
        )
        print(times)
        return probnum_solution

    def postprocess(self, odesol):
        return odesol
