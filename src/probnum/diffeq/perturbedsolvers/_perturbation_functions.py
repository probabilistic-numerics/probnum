"""perturbation functions to perturb the stepsize.

References
----------
.. [1] https://arxiv.org/abs/1801.01340
"""
import numpy as np
import scipy


def perturb_uniform(step, solver_order, noise_scale, random_state=None, size=()):
    """Perturb the step with uniformly distributed noise scaled by noise-scale [1]_.
    proposed by Abdulle and Garegnani(2020)

    Parameters
    ----------
    step : float
        unperturbed step propesed by the steprule
    solver_order : int
        order of the solver
    noise_scale : float
        scales the perturbation
    seed : int
        seed for pseudo-random number generator.
    """
    scipy.stats.uniform.rvs(random_state=random_state)
    try:
        noisy_step = np.random.uniform(
            step - noise_scale * step ** (solver_order + 0.5),
            step + noise_scale * step ** (solver_order + 0.5),
            size,
        )
    except ValueError:
        print("ValueError: Stepsize is too large (>=1)")
    return noisy_step


def perturb_lognormal(step, solver_order, noise_scale, random_state=None, size=()):
    """Perturb the step with lognormally distributed noise scaled by noise-scale [1]_.

    Parameters
    ----------
    step : float
        unperturbed step propesed by the steprule
    solver_order : int
        order of the solver
    noise_scale : float
        scales the perturbation
    seed : int
        seed for pseudo-random number generator.
    """
    scipy.stats.uniform.rvs(random_state=random_state)
    mean = np.log(step) - np.log(
        np.sqrt(1 + noise_scale * (step ** (2 * solver_order)))
    )
    cov = np.log(1 + noise_scale * (step ** (2 * solver_order)))
    noisy_step = np.exp(np.random.normal(mean, cov, size))
    return noisy_step
