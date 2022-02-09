"""Utility functions for the interfaces."""

from probnum.diffeq import stepsize


def construct_steprule(*, ivp, adaptive, step, atol, rtol):
    if adaptive is True:
        if atol is None or rtol is None:
            raise ValueError(
                "Please provide absolute and relative tolerance for adaptive steps."
            )
        firststep = step if step is not None else stepsize.propose_firststep(ivp)
        steprule = stepsize.AdaptiveSteps(firststep=firststep, atol=atol, rtol=rtol)
    else:
        if step is None:
            raise ValueError("Constant steps require a 'step' argument.")
        steprule = stepsize.ConstantSteps(step)
    return steprule
