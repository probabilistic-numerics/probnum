"""Update the belief over the parameters with observations in a 1D quadratic
optimization problem."""

from typing import Union

import numpy as np

import probnum as pn
from probnum import linops, randvars
from probnum.typing import FloatArgType


def gaussian_belief_update(
    fun_params0: randvars.RandomVariable,
    action: FloatArgType,
    observation: FloatArgType,
    noise_cov: Union[np.ndarray, linops.LinearOperator],
) -> randvars.RandomVariable:
    """Update the belief over the parameters with an observation.

    Parameters
    ----------
    fun_params0 :
        Belief over the parameters of the quadratic objective.
    action :
        Action of the probabilistic quadratic optimizer.
    observation :
        Observation of the problem corresponding to the given `action`.
    noise_cov :
        *shape=(3, 3)* -- Covariance of the noise on the parameters of the quadratic
        objective given by the assumed observation model.
    """
    # Feature vector
    x = np.asarray(action).reshape(1, -1)
    Phi = np.vstack((0.5 * x ** 2, x, np.ones_like(x)))

    # Mean and covariance
    mu = fun_params0.mean
    Sigma = fun_params0.cov

    # Gram matrix
    gram = Phi.T @ (Sigma + noise_cov) @ Phi

    # Posterior Mean
    m = mu + Sigma @ Phi @ np.linalg.solve(gram, observation - Phi.T @ mu)

    # Posterior Covariance
    S = Sigma - Sigma @ Phi @ np.linalg.solve(gram, Phi.T @ Sigma)

    return randvars.Normal(mean=m, cov=S)
