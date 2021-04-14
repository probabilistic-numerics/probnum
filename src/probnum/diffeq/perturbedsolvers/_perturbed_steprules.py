"""Noisy steprules for ODE solvers."""
# -*- coding: utf-8 -*-
import numpy as np

from probnum import diffeq


class NoisyStepRule(diffeq.StepRule):
    """
    Examples
    --------
    >>> perturb_function = functools.partial(parturb_lognormal, noise_scale=2.)
    >>> steprule = NoisyStepRule(steprule, perturb_function)
    """

    def __init__(self, steprule, perturb_function):
        self.steprule = steprule
        self.perturb_function = perturb_function
        self.firststep = self.perturb_function(self.steprule.firststep)
        super().__init__(firststep=self.firststep)

    def is_accepted(self, scaled_error):
        return self.steprule.is_accepted(scaled_error)

    def suggest(self, args):
        step = self.steprule.suggest(args)
        return self.perturb_function(step)

    def errorest_to_norm(self, errorest, proposed_state, current_state):
        return self.steprule.errorest_to_norm(errorest, proposed_state, current_state)


class UniformNoisyStepRule(diffeq.StepRule):
    """Step size adaptation as proposed by Abdulle et al. The stepsizes are uniformly
    perturbed with random noise that depends on the order of the solver.

    Parameters
    ----------
    steprule : :obj:'StepRule'
        can be one of AdaptiveSteps or ConstantSteps
    order : float, optional
        corresponds to a noise-scale, convergence guarantee for order = order of the solver
    """

    def __init__(self, steprule, order, noise_scale):
        self.steprule = steprule
        self.order = order
        self.noise_scale = noise_scale
        first_noisy_step = self.perturb(self.steprule.firststep)
        super().__init__(firststep=first_noisy_step)

    def is_accepted(self, scaled_error):
        return True

    def suggest(self, laststep, errorest, **kwargs):
        step = self.steprule.suggest(laststep, errorest, **kwargs)
        # step>=1 doesn't fulfill convergence criteria
        new_step = self.perturb(step)
        return new_step

    def errorest_to_norm(self, errorest, proposed_state, current_state):
        return self.steprule.errorest_to_norm(errorest, proposed_state, current_state)

    def perturb_uniform(self, step):
        if step < 1:
            new_step = np.random.uniform(
                step - self.noise_scale * step ** (self.order + 0.5),
                step + self.noise_scale * step ** (self.order + 0.5),
            )
        else:
            print("Error: Stepsize too large (>=1), not possible")
        return new_step


class LogNormalNoisyStepRule(diffeq.StepRule):
    """Step size adaptation as proposed by Abdulle et al. The stepsizes are lognormally
    perturbed with random noise that depends on the order of the solver.

    Parameters
    ----------
    steprule : :obj:'StepRule'
        can be one of AdaptiveSteps or ConstantSteps
    order : float, optional
        corresponds to a noise-scale, convergence guarantee for order = order of the solver
    """

    def __init__(self, steprule, order, noise_scale):
        self.steprule = steprule
        self.order = order
        self.noise_scale = noise_scale
        first_noisy_step = self.perturb(self.steprule.firststep)
        super().__init__(firststep=first_noisy_step)

    def is_accepted(self, scaled_error):
        return True

    def suggest(self, laststep, errorest, **kwargs):
        step = self.steprule.suggest(laststep, errorest, **kwargs)
        new_step = self.perturb(step)
        return new_step

    def errorest_to_norm(self, errorest, proposed_state, current_state):
        return self.steprule.errorest_to_norm(errorest, proposed_state, current_state)

    def perturb_lognormal(self, step):
        mean = np.log(step) - np.log(
            np.sqrt(1 + self.noise_scale * (step ** (2 * self.order)))
        )
        cov = np.log(1 + self.noise_scale * (step ** (2 * self.order)))
        step = np.exp(np.random.normal(mean, cov))
        return step
