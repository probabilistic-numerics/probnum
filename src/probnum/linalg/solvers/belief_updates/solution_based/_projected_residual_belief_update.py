"""Belief update in a solution-based inference view where the information is given by
projecting the current residual to a subspace."""
import numpy as np

import probnum  # pylint: disable="unused-import"
from probnum import randvars
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.typing import FloatLike

from .._linear_solver_belief_update import LinearSolverBeliefUpdate


class ProjectedResidualBeliefUpdate(LinearSolverBeliefUpdate):
    r"""Gaussian belief update given projected residual information.

    Updates the belief over the quantities of interest of a linear system :math:`Ax=b`
    given a Gaussian belief over the solution :math:`x` and information of the form
    :math:`s\^top r_i = s^\top (b - Ax_i) = s^\top A (x - x_i)`.
    The belief update computes the posterior belief about the solution, given by
    :math:`p(x \mid y) = \mathcal{N}(x; x_{i+1}, \Sigma_{i+1})`, such that

    .. math ::
        \begin{align}
            x_{i+1} &= x_i + \Sigma_i A^\top s (s^\top A \Sigma_i A^\top s +
            \lambda)^\dagger s^\top r_i,\\
            \Sigma_{i+1} &= \Sigma_i - \Sigma_i A^\top s (s^\top A \Sigma_i A s +
            \lambda)^\dagger s^\top A \Sigma_i,
        \end{align}

    where :math:`\lambda` is the noise variance.


    Parameters
    ----------
    noise_var :
        Variance of the scalar observation noise.
    """

    def __init__(self, noise_var: FloatLike = 0.0) -> None:
        if noise_var < 0.0:
            raise ValueError(f"Noise variance {noise_var} must be non-negative.")
        self._noise_var = noise_var

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> LinearSystemBelief:

        proj_resid = solver_state.observation

        # Compute gain and covariance update
        action_A = solver_state.action.T @ solver_state.problem.A
        cov_xy = solver_state.belief.x.cov @ action_A.T
        gram = action_A @ cov_xy + self.noise_var
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

    @property
    def noise_var(self) -> float:
        """Observation noise."""
        return self._noise_var
