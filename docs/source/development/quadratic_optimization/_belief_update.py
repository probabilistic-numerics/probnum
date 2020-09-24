from typing import Union, Callable
from probnum.type import FloatArgType

import numpy as np
import probnum as pn
import probnum.random_variables as rvs
import probnum.linalg.linops as linops


class QuadOptBeliefUpdate:
    """
    Update the belief over the parameters with observations.

    Parameters
    ----------
    belief_update :
        Belief update over the parameters of the latent quadratic function given an
        action-observation pair.
    """

    def __init__(
        self,
        belief_update: Callable[
            [
                pn.RandomVariable,
                FloatArgType,
                FloatArgType,
            ],
            pn.RandomVariable,
        ],
    ):
        self._belief_update = belief_update

    def __call__(
        self,
        fun_params0: pn.RandomVariable,
        action: FloatArgType,
        observation: FloatArgType,
    ) -> pn.RandomVariable:
        """
        Update the belief over the parameters with an observation.

        Parameters
        ----------
        fun_params0 :
            Belief over the parameters of the quadratic objective.
        action :
            Action of the probabilistic quadratic optimizer.
        observation :
            Observation of the problem corresponding to the given `action`.
        """
        return self._belief_update(fun_params0, action, observation)


class GaussianBeliefUpdate(QuadOptBeliefUpdate):
    """
    Update the belief over the parameters with observations assuming a Gaussian noise
    model for the observations.

    Parameters
    ----------
    noise_cov :
        *shape=(3, 3)* -- Covariance of the noise on the parameters of the quadratic
        objective given by the assumed observation model.
    """

    def __init__(self, noise_cov: Union[np.ndarray, linops.LinearOperator]):
        self.noise_cov = noise_cov
        super().__init__(belief_update=self._gaussian_belief_update)

    def _gaussian_belief_update(
        self,
        fun_params0: pn.RandomVariable,
        action: FloatArgType,
        observation: FloatArgType,
    ) -> pn.RandomVariable:
        """
        Update the belief over the parameters with an observation.

        Parameters
        ----------
        fun_params0 :
            Belief over the parameters of the quadratic objective.
        action :
            Action of the probabilistic quadratic optimizer.
        observation :
            Observation of the problem corresponding to the given `action`.
        """
        # Feature vector
        x = np.asarray(action).reshape(1, -1)
        Phi = np.vstack((0.5 * x ** 2, x, np.ones_like(x)))

        # Mean and covariance
        mu = fun_params0.mean
        Sigma = fun_params0.cov

        # Gram matrix
        gram = Phi.T @ (Sigma + self.noise_cov) @ Phi

        # Posterior Mean
        m = mu + Sigma @ Phi @ np.linalg.solve(gram, observation - Phi.T @ mu)

        # Posterior Covariance
        S = Sigma - Sigma @ Phi @ np.linalg.solve(gram, Phi.T @ Sigma)

        return rvs.Normal(mean=m, cov=S)
