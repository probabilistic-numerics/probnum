from abc import ABC, abstractmethod


class StepRule(ABC):
    """(Adaptive) step size rules for ODE solvers."""

    def __init__(self, firststep):
        self.firststep = firststep

    @abstractmethod
    def suggest(self, laststep, errorest, localconvrate=None):
        """Suggest a new step h_{n+1} given error estimate e_n at step h_n."""
        raise NotImplementedError

    @abstractmethod
    def is_accepted(self, proposedstep, errorest, localconvrate=None):
        """Check if the proposed step should be accepted or not.

        Variable "proposedstep" not used yet, but may be important in
        the future, e.g. if we decide that instead of tol_per_step (see
        AdaptiveSteps) we want to be able to control tol_per_unitstep.
        """
        raise NotImplementedError


class ConstantSteps(StepRule):
    """Constant step size rule for ODE solvers."""

    def __init__(self, stepsize):
        self.step = stepsize
        super().__init__(firststep=stepsize)

    def suggest(self, laststep, errorest, localconvrate=None):
        return self.step

    def is_accepted(self, proposedstep, errorest, localconvrate=None):
        """Meaningless since always True."""
        return True


class AdaptiveSteps(StepRule):
    """Adaptive step size selection based on tolerance per step.

    By default, there is no being "too small" for a step. However, a
    Warning is printed if the suggested step is smaller than roughly
    machine precision: 1e-15.

    Parameters
    ----------
    tol_per_step : float
        Tolerance per step (absolute)
    limitchange : list with 2 elements, optional
        Lower and upper bounds for computed change of step.
    safetyscale : float, optional
        Safety factor for proposal of distributions, 0 << safetyscale < 1
    """

    def __init__(
        self, tol_per_step, firststep, limitchange=(0.1, 5.0), safetyscale=0.95
    ):
        self.tol_per_step = float(tol_per_step)
        self.safetyscale = float(safetyscale)
        self.limitchange = limitchange
        super().__init__(firststep=firststep)

    def suggest(self, laststep, errorest, localconvrate=None):
        small, large = self.limitchange
        ratio = self.tol_per_step / (laststep * errorest)
        change = self.safetyscale * ratio ** (1.0 / localconvrate)
        if change < small:
            step = small * laststep
        elif large < change:
            step = large * laststep
        else:
            step = change * laststep
        if step < 1e-15:
            print("Warning: Stepsize is num. zero (h=%.1e)" % step)
        return step

    def is_accepted(self, proposedstep, errorest, localconvrate=None):
        return errorest * proposedstep < self.tol_per_step
