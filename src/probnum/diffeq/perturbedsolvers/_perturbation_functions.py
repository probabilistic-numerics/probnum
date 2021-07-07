"""Perturbation functions to perturb the stepsize."""
import numpy as np
import scipy


def perturb_uniform(step, solver_order, noise_scale, random_state=None, size=()):
    """Perturb the step with uniformly distributed noise scaled by noise-scale.

    based on the work by Abdulle and Garegnani [1]_ .

    Parameters
    ----------
    step : float
        unperturbed step propesed by the steprule
    solver_order : int
        order of the solver
    noise_scale : float
        scales the perturbation
    seed : RandomStateArgType
        seed for pseudo-random number generator.

    References
    ----------
    .. [1] Abdulle, A., Garegnani, G. (2020). Random time step probabilistic methods for
       uncertainty quantification in chaotic and geometric numerical integration.
       Statistics and Computing, 1-26.
    """
    if step >= 1.0:
        raise ValueError("Stepsize too large (>= 1)")
    else:
        uniform_rv_samples = scipy.stats.uniform.rvs(
            random_state=random_state, size=size
        )
        shift = noise_scale * step ** (solver_order + 0.5)
        left_boundary = step - shift
        right_boundary = step + shift
        samples = left_boundary + (right_boundary - left_boundary) * uniform_rv_samples
    return samples


def perturb_lognormal(step, solver_order, noise_scale, random_state=None, size=()):
    """Perturb the step with lognormally distributed noise scaled by noise-scale.

    based on the work by Abdulle and Garegnani [1]_ .

    Parameters
    ----------
    step : float
        unperturbed step propesed by the steprule
    solver_order : int
        order of the solver
    noise_scale : float
        scales the perturbation
    seed : RandomStateArgType
        seed for pseudo-random number generator.

    References
    ----------
    .. [1] Abdulle, A., Garegnani, G. (2020). Random time step probabilistic methods for
       uncertainty quantification in chaotic and geometric numerical integration.
       Statistics and Computing, 1-26.
    """
    shift = 0.5 * np.log(1 + noise_scale * (step ** (2 * solver_order)))
    mean = np.log(step) - shift
    cov = 2 * shift
    samples = np.exp(
        scipy.stats.multivariate_normal.rvs(
            mean=mean, cov=cov, size=size, random_state=random_state
        )
    )
    return samples
