from abc import ABC, abstractmethod

import numpy as np


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

    # 'internalnorm' is named after the respective variable used by SciML.
    @abstractmethod
    def errorest_to_internalnorm(self, errorest, proposed_rv, current_rv, atol, rtol):
        """Computes the internal norm (usually referred to as 'E').

        The internal norm is usually the current error estimate
        normalised with atol, rtol, and the magnitude of the previous
        states.
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

    def errorest_to_internalnorm(self, errorest, proposed_rv, current_rv, atol, rtol):
        pass


# Once we have other controls, e.g. PI control, we can rename this into ProportionalControl.
# Until then, lets keep the delta small, I'd say (N).
class AdaptiveSteps(StepRule):
    """Adaptive step size selection using proportional control.

    Parameters
    ----------
    firststep : float
        First step to be taken by the ODE solver (which happens in absence of error estimates).
    limitchange : list with 2 elements, optional
        Lower and upper bounds for computed change of step.
    safetyscale : float, optional
        Safety factor for proposal of distributions, 0 << safetyscale < 1
    minstep : float, optional
        Minimum step that is allowed. A runtime error is thrown if the proposed step is smaller. Default is 1e-15.
    maxstep : float, optional
        Maximum step that is allowed. A runtime error is thrown if the proposed step is larger. Default is 1e15.
    """

    def __init__(
        self,
        firststep,
        limitchange=(0.1, 5.0),
        safetyscale=0.95,
        minstep=1e-15,
        maxstep=1e15,
    ):
        self.safetyscale = float(safetyscale)
        self.limitchange = limitchange
        self.minstep = minstep
        self.maxstep = maxstep

        super().__init__(firststep=firststep)

    def suggest(self, laststep, scaled_error, localconvrate=None):
        small, large = self.limitchange

        ratio = 1.0 / scaled_error
        change = self.safetyscale * ratio ** (1.0 / localconvrate)

        # The below code should be doable in a single line?
        if change < small:
            step = small * laststep
        elif large < change:
            step = large * laststep
        else:
            step = change * laststep

        if step < self.minstep:
            raise RuntimeError("Step-size smaller than minimum step-size")
        if step > self.maxstep:
            raise RuntimeError("Step-size larger than maximum step-size")
        return step

    # Looks unnecessary, though maybe we want tolerance per unit step
    # in which case it is good to have such method in here.
    def is_accepted(self, laststep, scaled_error, localconvrate=None):
        return scaled_error < 1

    # In here, because (i) we do not want to compute it for constant steps,
    # and in fact, we don't even want to think about which value atol and rtol should have;
    # (ii) having it in StepRule makes it easier to test, because the class is more light-weight;
    # (iii) who knows, maybe there are other ways of dealing with this.
    def errorest_to_internalnorm(
        self, errorest, proposed_state, current_state, atol, rtol
    ):
        tolerance = atol + rtol * np.maximum(
            np.abs(proposed_state), np.abs(current_state)
        )
        ratio = errorest / tolerance
        dim = len(ratio) if ratio.ndim > 0 else 1
        return np.linalg.norm(ratio) / np.sqrt(dim)
