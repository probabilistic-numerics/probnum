"""ODE-Solver as proposed by Abdulle and Garegnani."""

from typing import Callable

import numpy as np

from probnum import _randomvariablelist, randvars
from probnum.typing import FloatArgType

from ..odesolver import ODESolver
from ..wrappedscipysolver import WrappedScipyRungeKutta
from ._perturbedstepsolution import PerturbedStepSolution


class PerturbedStepSolver(ODESolver):
    """ODE-Solver random perturbatino of the step-sizes.

    Perturbs the steps accordingly and projects the solution back to the originally
    proposed time points. Proposed by Abdulle and Garegnani (2020) [1]_.

    Parameters
    ----------
    rng :
        Random number generator.
    solver :
        Currently this has to be a Runge-Kutta method based on SciPy.
    noise-scale :
        Scales the amount of noise that is introduced.
    perturb_function :
        Defines how the stepsize is distributed. This can be either one of
        ``perturb_lognormal()`` or ``perturb_uniform()`` or any other perturbation function with
        the same signature.

    References
    ----------
    .. [1] Abdulle, A. and Garegnani, G.
        Random time step probabilistic methods for uncertainty quantification in chaotic and geometric numerical integration.
        Statistics and Computing. 2020.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        solver: WrappedScipyRungeKutta,
        noise_scale: FloatArgType,
        perturb_function: Callable,
    ):
        def perturb_step(rng, step):
            return perturb_function(
                rng=rng,
                step=step,
                solver_order=solver.order,
                noise_scale=noise_scale,
                size=(),
            )

        self.rng = rng
        self.perturb_step = perturb_step
        self.solver = solver
        self.scales = None
        super().__init__(ivp=solver.ivp, order=solver.order)

    def initialise(self):
        """Initialise and reset the solver."""
        self.scales = []
        return self.solver.initialise()

    def step(
        self, start: FloatArgType, stop: FloatArgType, current: randvars, **kwargs
    ):
        """Perturb the original stopping point.

        Perform one perturbed step and project the solution back to the original
        stopping point.

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
        noisy_step = self.perturb_step(self.rng, dt)
        state_as_rv, error_estimation, reference_state = self.solver.step(
            start, start + noisy_step, current
        )
        scale = noisy_step / dt
        self.scales.append(scale)
        return state_as_rv, error_estimation, reference_state

    def method_callback(self, time, current_guess, current_error):
        """Call dense output after each step and store the interpolants."""
        return self.solver.method_callback(time, current_guess, current_error)

    def rvlist_to_odesol(
        self, times: np.ndarray, rvs: _randomvariablelist._RandomVariableList
    ):
        interpolants = self.solver.interpolants
        probnum_solution = PerturbedStepSolution(self.scales, times, rvs, interpolants)
        return probnum_solution

    def postprocess(self, odesol):
        return self.solver.postprocess(odesol)
