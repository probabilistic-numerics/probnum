"""Perturbation functions to perturb the stepsize.

References
----------
.. [1] https://arxiv.org/abs/1801.01340
"""
from typing import Optional, Union

import numpy as np
import scipy

from probnum.typing import FloatArgType, IntArgType, ShapeArgType


def perturb_uniform(
    rng: np.random.Generator,
    step: FloatArgType,
    solver_order: IntArgType,
    noise_scale: FloatArgType,
    size: Optional[ShapeArgType] = (),
) -> Union[float, np.ndarray]:
    """Perturb the step with uniformly distributed noise scaled by noise-scale [1]_.

    Proposed by Abdulle and Garegnani (2020).

    Parameters
    ----------
    rng
        Random number generator
    step
        Unperturbed step propesed by the steprule
    solver_order
        Order of the solver
    noise_scale
        Scales the perturbation
    size
        Number of perturbation samples to be drawn. Optional. Default is ``size=()``.
    """
    if step >= 1.0:
        raise ValueError("Stepsize too large (>= 1)")
    else:
        uniform_rv_samples = scipy.stats.uniform.rvs(random_state=rng, size=size)
        shift = noise_scale * step ** (solver_order + 0.5)
        left_boundary = step - shift
        right_boundary = step + shift
        samples = left_boundary + (right_boundary - left_boundary) * uniform_rv_samples
    return samples


def perturb_lognormal(
    rng: np.random.Generator,
    step: FloatArgType,
    solver_order: IntArgType,
    noise_scale: FloatArgType,
    size: Optional[ShapeArgType] = (),
) -> Union[float, np.ndarray]:
    """Perturb the step with lognormally distributed noise scaled by noise-scale [1]_.

    Parameters
    ----------
    rng
        Random number generator
    step
        Unperturbed step propesed by the steprule
    solver_order
        Order of the solver
    noise_scale
        Scales the perturbation
    size
        Number of perturbation samples to be drawn. Optional. Default is ``size=()``.
    """
    shift = 0.5 * np.log(1 + noise_scale * (step ** (2 * solver_order)))
    mean = np.log(step) - shift
    cov = 2 * shift
    samples = np.exp(
        scipy.stats.multivariate_normal.rvs(
            mean=mean, cov=cov, size=size, random_state=rng
        )
    )
    return samples
