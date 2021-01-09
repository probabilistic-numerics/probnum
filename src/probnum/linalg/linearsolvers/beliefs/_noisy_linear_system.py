"""Belief over a linear system with noise-corrupted system matrix."""

from typing import List, Optional

import numpy as np

import probnum
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers.beliefs._linear_system_belief import (
    LinearSystemBelief,
)
from probnum.linalg.linearsolvers.hyperparam_optim import OptimalNoiseScale
from probnum.problems import LinearSystem


class NoisyLinearSystemBelief(LinearSystemBelief):
    r"""Belief over a noise-corrupted linear system.

    Parameters
    ----------
    x :
        Belief over the solution.
    A :
        Belief over the system matrix.
    Ainv :
        Belief over the (pseudo-)inverse of the system matrix.
    b :
        Belief over the right hand side.
    noise_scale :
        Estimate for the scale of the noise on the system matrix.
    """

    def __init__(
        self,
        x: rvs.RandomVariable,
        A: rvs.RandomVariable,
        Ainv: rvs.RandomVariable,
        b: rvs.RandomVariable,
        noise_scale: float = None,
    ):
        self.noise_scale = noise_scale
        super().__init__(x=x, A=A, Ainv=Ainv, b=b)

    def optimize_hyperparams(
        self,
        problem: LinearSystem,
        actions: List[np.ndarray],
        observations: List[np.ndarray],
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Optional["probnum.linalg.linearsolvers.LinearSolverState"]:
        """Estimate the noise level of a noisy linear system.

        Computes the optimal noise scale maximizing the log-marginal
        likelihood.
        """
        self.noise_scale, solver_state = OptimalNoiseScale()(
            problem=problem,
            belief=self,
            actions=actions,
            observations=observations,
            solver_state=solver_state,
        )
        return solver_state
