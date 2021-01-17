"""Abstract ODESolver class.

Interface for Runge-Kutta, ODEFilter.
"""

from abc import ABC, abstractmethod

import probnum.random_variables as pnrv

from .odesolution import ODESolution


class ODESolver(ABC):
    """Interface for ODE solvers."""

    def __init__(self, ivp, order):
        self.ivp = ivp
        self.order = order  # e.g.: RK45 has order=5, IBM(q) has order=q
        self.num_steps = 0  # move to ODESolution?

    def solve(self, steprule):
        """Solve an IVP.

        Parameters
        ----------
        steprule : :class:`StepRule`
            Step-size selection rule, e.g. constant steps or adaptive steps.
        """
        odesol, t, current_rv = self.initialize()  # "almost empty" ODE solution
        odesol.append(t, current_rv)
        stepsize = steprule.firststep

        while t < self.ivp.tmax:

            t_new = t + stepsize
            proposed_rv, errorest = self.step(t, t_new, current_rv)
            internal_norm = steprule.errorest_to_norm(
                errorest=errorest,
                proposed_state=proposed_rv.mean,
                current_state=current_rv.mean,
            )
            if steprule.is_accepted(internal_norm):
                self.num_steps += 1
                self.method_callback(
                    time=t_new, current_guess=proposed_rv, current_error=errorest
                )
                t = t_new
                current_rv = proposed_rv
                odesol.append(t, current_rv)

            suggested_stepsize = steprule.suggest(
                stepsize, internal_norm, localconvrate=self.order + 1
            )
            stepsize = min(suggested_stepsize, self.ivp.tmax - t)

        odesol = self.postprocess(odesol)
        return odesol

    @abstractmethod
    def initialize(self) -> (ODESolution, float, pnrv.RandomVariable):
        """Return an empty ODESolution object as well as suitable t0 and y0.

        Required for implementation of ODE solvers.

        These values might be different for different solvers.
        For instance, the y0 that is used for the ODE filters, is a stack
        (y0, dy0, ddy0, ...) and the ODESolution is a KalmanODESolution object.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, start: float, stop: float, current: pnrv.RandomVariable, **kwargs
    ) -> (pnrv.RandomVariable, float):
        """Implement an ODE solver by implementing a step() method."""
        raise NotImplementedError

    def postprocess(self, odesol: ODESolution) -> ODESolution:
        """Process the ODESolution object before returning.

        Optional.
        """
        return odesol

    def method_callback(
        self, time: float, current_guess: pnrv.RandomVariable, current_error: float
    ) -> None:
        """Callback that is carried out after accepting the current random variable but
        before storing it."""
        pass
