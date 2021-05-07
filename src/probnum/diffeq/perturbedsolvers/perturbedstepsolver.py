"""ODE solver as proposed by Abdulle and Garegnani."""
import numpy as np

from probnum import _randomvariablelist, diffeq, randvars, utils
from probnum.type import FloatArgType


class PerturbedStepSolver(diffeq.ODESolver):
    """ODE Solver based on Scipy that introduces uncertainty by perturbing the time-
    steps."""

    def __init__(
        self,
        # diffeq.WrappedScipyRungeKutta
        solver,
        noise_scale: FloatArgType,
        perturb_function,
        random_state=None,
    ):
        def perturbation_step(step):
            return perturb_function(
                step=step,
                solver_order=solver.order,
                noise_scale=noise_scale,
                random_state=utils.as_random_state(random_state),
                size=(),
            )

        self.perturb_step = perturbation_step
        self.solver = solver
        self.time = None
        self.scale = None
        self.scales = None
        super().__init__(ivp=solver.ivp, order=solver.order)

    def initialise(self):
        """Initialise and reset the solver."""
        self.scales = []
        return self.solver.initialise()

    def step(
        self, start: FloatArgType, stop: FloatArgType, current: randvars, **kwargs
    ):
        """Perturb the original stopping point, perform one perturbed step and project
        the solution back to the original stopping point.

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

    def rvlist_to_odesol(
        self, times: np.ndarray, rvs: _randomvariablelist._RandomVariableList
    ):
        interpolants = self.solver.interpolants
        probnum_solution = diffeq.PerturbedStepSolution(
            self.scales, times, rvs, interpolants
        )
        return probnum_solution

    def postprocess(self, odesol):
        return odesol
