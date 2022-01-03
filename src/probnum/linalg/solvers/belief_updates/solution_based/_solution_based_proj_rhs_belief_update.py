"""Belief update in a solution-based inference view where the information is given by
projecting the current residual to a subspace."""
import numpy as np

import probnum  # pylint: disable="unused-import"
from probnum import randvars
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.typing import FloatLike

from .._linear_solver_belief_update import LinearSolverBeliefUpdate


class SolutionBasedProjectedRHSBeliefUpdate(LinearSolverBeliefUpdate):
    r"""Gaussian belief update in a solution-based inference framework assuming projected right-hand-side information.

    Updates the belief over the quantities of interest of a linear system :math:`Ax=b` given a Gaussian belief over the solution :math:`x` and information of the form :math:`y = s\^top b=s^\top Ax`. The belief update computes the posterior belief about the solution, given by :math:`p(x \mid y) = \mathcal{N}(x; x_{i+1}, \Sigma_{i+1})`, [1]_ such that

    .. math ::
        \begin{align}
            x_{i+1} &= x_i + \Sigma_i A^\top s (s^\top A \Sigma_i A^\top s + \lambda)^\dagger s^\top (b - Ax_i),\\
            \Sigma_{i+1} &= \Sigma_i - \Sigma_i A^\top s (s^\top A \Sigma_i A s + \lambda)^\dagger s^\top A \Sigma_i,
        \end{align}

    where :math:`\lambda` is the noise variance.


    Parameters
    ----------
    noise_var :
        Variance of the scalar observation noise.

    References
    ----------
    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian
       Analysis*, 2019, 14, 937-1012
    """

    def __init__(self, noise_var: FloatLike = 0.0) -> None:
        if noise_var < 0.0:
            raise ValueError(f"Noise variance {noise_var} must be non-negative.")
        self._noise_var = noise_var

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> LinearSystemBelief:

        action_A = solver_state.action @ solver_state.problem.A
        pred = action_A @ solver_state.belief.x.mean
        proj_resid = solver_state.observation - pred
        cov_xy = solver_state.belief.x.cov @ action_A.T
        gram = action_A @ cov_xy + self._noise_var
        gram_pinv = 1.0 / gram if gram > 0.0 else 0.0
        gain = cov_xy * gram_pinv
        cov_update = np.outer(gain, cov_xy)

        x = randvars.Normal(
            mean=solver_state.belief.x.mean + gain * proj_resid,
            cov=solver_state.belief.x.cov - cov_update,
        )
        if solver_state.belief.Ainv is None:
            Ainv = randvars.Constant(cov_update)
        else:
            Ainv = solver_state.belief.Ainv + cov_update

        return LinearSystemBelief(
            x=x, A=solver_state.belief.A, Ainv=Ainv, b=solver_state.belief.b
        )
