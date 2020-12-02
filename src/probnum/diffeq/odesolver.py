"""Abstract ODESolver class.

Interface for Runge-Kutta, ODEFilter.
"""

from abc import ABC, abstractmethod

from probnum.diffeq.odesolution import ODESolution


class ODESolver(ABC):
    """Interface for ODESolver."""

    def __init__(self, ivp, order):
        self.ivp = ivp
        self.order = order  # RK45 has order=5, IBM(q) has order=q
        self.num_steps = 0

    def solve(self, steprule, atol, rtol, **kwargs):
        """Solve an IVP.

        Parameters
        ----------
        firststep : float
            First step for adaptive step-size rule.
        steprule : :class:`StepRule`
            Step-size selection rule, e.g. constant steps or adaptive steps.
        """
        t, current_rv = self.initialise()
        times, rvs = [t], [current_rv]
        stepsize = steprule.firststep

        while t < self.ivp.tmax:

            t_new = t + stepsize
            proposed_rv, errorest = self.step(t, t_new, current_rv, **kwargs)
            internal_norm = steprule.errorest_to_internalnorm(
                errorest=errorest,
                proposed_rv=proposed_rv,
                current_rv=current_rv,
                atol=atol,
                rtol=rtol,
            )
            if steprule.is_accepted(stepsize, internal_norm):
                self.num_steps += 1
                self.method_callback(
                    time=t_new, current_guess=proposed_rv, current_error=errorest
                )
                t = t_new
                current_rv = proposed_rv
                times.append(t)
                rvs.append(current_rv)

            suggested_stepsize = steprule.suggest(
                stepsize, internal_norm, localconvrate=self.order + 1
            )
            stepsize = min(suggested_stepsize, self.ivp.tmax - t)

        odesol = self.postprocess(times=times, rvs=rvs)
        return odesol

    @abstractmethod
    def initialise(self):
        """Returns t0 and y0 (for the solver, which might be different to ivp.y0)"""
        raise NotImplementedError

    @abstractmethod
    def step(self, start, stop, current, **kwargs):
        """Every ODE solver needs a step() method that returns a new random variable and
        an error estimate."""
        raise NotImplementedError

    def postprocess(self, times, rvs):
        """Turn list of random variables into an ODE solution object and potentially do
        more. Overwrite for instance via.

        >>> def postprocess(self, times, rvs):
        >>> # do something with times and rvs
        >>> odesol = super().postprocess(times, rvs)
        >>> # do something with odesol
        >>> return odesol
        """
        return ODESolution(times, rvs, self)

    def method_callback(self, time, current_guess, current_error):
        """Optional callback.

        Can be overwritten. Do this as soon as it is clear that the
        current guess is accepted, but before storing it. No return. For
        example: tune hyperparameters (sigma).
        """
        pass
