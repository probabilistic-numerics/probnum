"""ODE-Solver as proposed by Abdulle and Garegnani."""

from typing import Callable, Optional

import numpy as np

from probnum import _randomvariablelist, diffeq, randvars, utils
from probnum.diffeq.wrappedscipysolver import WrappedScipyRungeKutta
from probnum.typing import FloatArgType, RandomStateArgType


class PerturbedStepSolver(diffeq.ODESolver):
    """ODE-Solver based on Abdulle and Garegnani.

    Perturbs the steps accordingly and projects the solution back to the originally
    proposed time points.

    Parameters
    ----------
    solver :
        Currently this has to be a Runge-Kutta method based on SciPy.
    noise-scale :
        Scales the amount of noise that is introduced.
    perturb_function :
        Defines how the stepsize is distributed. This can be either one of
        ``perturb_lognormal()`` or ``perturb_uniform()`` or any other perturbation function with
        input parameters step, ``solver_order``, ``noise_scale``, ``random_state`` and size.
    random_state :
        Random state (seed, generator) to be used for sampling base measure realizations.
    """

    def __init__(
        self,
        solver: WrappedScipyRungeKutta,
        noise_scale: FloatArgType,
        perturb_function: Callable,
        random_state: Optional[RandomStateArgType] = None,
    ):
        random_state = utils.as_random_state(random_state)

        def perturb_step(step):
            return perturb_function(
                step=step,
                solver_order=solver.order,
                noise_scale=noise_scale,
                random_state=random_state,
                size=(),
            )

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
        stopping point [1]_.

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

        References
        ----------
        .. [1] Abdulle, A., Garegnani, G. (2020). Random time step probabilistic methods for
           uncertainty quantification in chaotic and geometric numerical integration.
           Statistics and Computing, 1-26.
        """

        dt = stop - start
        noisy_step = self.perturb_step(dt)
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
        probnum_solution = diffeq.PerturbedStepSolution(
            self.scales, times, rvs, interpolants
        )
        return probnum_solution

    def postprocess(self, odesol):
        return self.solver.postprocess(odesol)
