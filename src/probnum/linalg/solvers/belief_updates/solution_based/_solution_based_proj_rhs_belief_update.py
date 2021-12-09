"""Belief update in a solution-based inference view where the information is given by
projecting the current residual to a subspace."""
from typing import Callable, Optional

import numpy as np

import probnum  # pylint: disable="unused-import"
from probnum import randvars
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.typing import FloatArgType
from probnum.utils.linalg import double_gram_schmidt

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
    noise_var
        Variance of the scalar observation noise.
    reorthogonalization_fn
        Reorthogonalization function, which takes a residual and a set of previous residuals to orthogonalize against.

    References
    ----------
    .. [1] Cockayne, J. et al., A Bayesian Conjugate Gradient Method, *Bayesian
       Analysis*, 2019, 14, 937-1012
    """

    def __init__(
        self,
        noise_var: FloatArgType = 0.0,
        reorthogonalization_fn: Optional[
            Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = None,
    ) -> None:
        if noise_var < 0.0:
            raise ValueError(f"Noise variance {noise_var} must be non-negative.")
        self._noise_var = noise_var
        self._reorthogonalization_fn = reorthogonalization_fn

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> LinearSystemBelief:

        # Compute projected residual
        proj_resid = (
            -solver_state.action @ solver_state.residual
        )  # s' b - s' A x_k = s' (b - A x_k) = s' r_k

        # Compute covariance update
        action_A = solver_state.action @ solver_state.problem.A  # s' A
        cov_xy = solver_state.belief.x.cov @ action_A.T  # Sigma A s
        gram = action_A @ cov_xy + self._noise_var  # G = s' A Sigma A s
        gram_pinv = 1.0 / gram if gram > 0.0 else 0.0  # Ginv = (s' A Sigma A s)^+
        gain = cov_xy * gram_pinv  # gain = Sigma A s (s' A Sigma A s)^+

        cov_update = np.outer(gain, cov_xy)  # Sigma A s (s' A Sigma A s)^+ s' A Sigma

        # Compute new mean and covariance
        x = randvars.Normal(
            mean=solver_state.belief.x.mean
            + gain * proj_resid,  # x_k + Sigma A s (s' A Sigma A s)^+ * s' r_k
            cov=solver_state.belief.x.cov - cov_update,
        )
        if solver_state.belief.Ainv is None:
            Ainv = randvars.Constant(cov_update)
        else:
            Ainv = solver_state.belief.Ainv + cov_update

        # Reorthogonalize residuals
        new_residual = solver_state.problem.A @ x.mean - solver_state.problem.b
        solver_state.residual = self._reorthogonalization_fn(
            new_residual, solver_state.residuals
        )

        return LinearSystemBelief(
            x=x, A=solver_state.belief.A, Ainv=Ainv, b=solver_state.belief.b
        )
