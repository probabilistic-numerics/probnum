"""Belief update in a solution-based inference view where the information is given by
projecting the current residual to a subspace."""
import probnum  # pylint: disable="unused-import"
from probnum import randvars
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.typing import FloatArgType

from ._linear_solver_belief_update import LinearSolverBeliefUpdate


class SolutionBasedProjectedRHSBeliefUpdate(LinearSolverBeliefUpdate):
    r"""Gaussian belief update in a solution-based inference framework assuming projected right-hand-side information.

    Updates the belief over the quantities of interest of a linear system :math:`Ax=b` given a Gaussian belief over the solution :math:`x` and information of the form :math:`y = b^\top s=(Ax)^\top s`. The belief update computes the posterior belief about the solution, given by :math:`p(x \mid y) = \mathcal{N}(x; x_{i+1}, \Sigma_{i+1})`, [1]_ such that

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

    def __init__(self, noise_var: FloatArgType = 0.0) -> None:
        if noise_var < 0.0:
            raise ValueError(f"Noise variance {noise_var} must be non-negative.")
        self._noise_var = noise_var

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> LinearSystemBelief:

        A_action = solver_state.problem.A @ solver_state.action
        pred = A_action.T @ solver_state.belief.x.mean
        resid = solver_state.observation - pred
        cov_xy = solver_state.belief.x.cov @ A_action
        gram = A_action.T @ cov_xy + self._noise_var
        gram_pinv = 1.0 / gram if gram > 0.0 else 0.0
        gain = cov_xy * gram_pinv
        cov_update = gain @ cov_xy.T

        x = randvars.Normal(
            mean=solver_state.belief.x.mean + gain * resid,
            cov=solver_state.belief.x.cov - cov_update,
        )
        Ainv = solver_state.belief.Ainv + cov_update

        return LinearSystemBelief(
            x=x, A=solver_state.belief.A, Ainv=Ainv, b=solver_state.belief.b
        )
