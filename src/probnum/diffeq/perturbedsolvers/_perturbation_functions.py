"""perturbation functions that can be used for the perturbed solvers."""
import numpy as np


def perturb_uniform(step, order, noise_scale, seed=1):
    """perturbs the step with uniformly distributed noise scaled by the chosen noise-
    scale."""

    np.random.mtrand.RandomState(seed=seed)
    if step < 1:
        noisy_step = np.random.uniform(
            step - noise_scale * step ** (order + 0.5),
            step + noise_scale * step ** (order + 0.5),
        )
    else:
        print("Error: Stepsize too large (>=1), not possible")
    return noisy_step


def perturb_lognormal(step, order, noise_scale, seed=1):
    """perturbs the step with lognormally distributed noise scaled by the chosen noise-
    scale."""
    np.random.mtrand.RandomState(seed=seed)
    mean = np.log(step) - np.log(np.sqrt(1 + noise_scale * (step ** (2 * order))))
    cov = np.log(1 + noise_scale * (step ** (2 * order)))
    noisy_step = np.exp(np.random.normal(mean, cov))
    return noisy_step
