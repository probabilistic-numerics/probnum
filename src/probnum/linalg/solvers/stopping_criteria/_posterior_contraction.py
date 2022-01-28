"""Stopping criterion based on the uncertainty about a quantity of interest."""

import numpy as np

import probnum  # pylint: disable="unused-import"
from probnum.typing import ScalarLike

from ._linear_solver_stopping_criterion import LinearSolverStoppingCriterion


class PosteriorContractionStoppingCriterion(LinearSolverStoppingCriterion):
    r"""Posterior contraction stopping criterion.

    Terminate when the uncertainty about the quantity of interest :math:`q` is
    sufficiently small, i.e. if :math:`\sqrt{\operatorname{tr}(\mathbb{Cov}(q))}
    \leq \max(\text{atol}, \text{rtol} \lVert b \rVert_2)`, where :math:`q` is either
    the solution :math:`x`, the system matrix :math:`A` or its inverse :math:`A^{-1}`.

    Parameters
    ----------
    qoi :
        Quantity of interest. One of ``{"x", "A", "Ainv"}``.
    atol :
        Absolute tolerance.
    rtol :
        Relative tolerance.
    """

    def __init__(
        self,
        qoi: str = "x",
        atol: ScalarLike = 10 ** -5,
        rtol: ScalarLike = 10 ** -5,
    ):
        self.qoi = qoi
        self.atol = probnum.utils.as_numpy_scalar(atol)
        self.rtol = probnum.utils.as_numpy_scalar(rtol)

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> bool:
        """Check whether the uncertainty about the quantity of interest is smaller than
        the specified tolerance.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        trace_cov_qoi = getattr(solver_state.belief, self.qoi).cov.trace()
        b_norm = np.linalg.norm(solver_state.problem.b, ord=2)

        return (
            np.abs(trace_cov_qoi) <= self.atol ** 2
            or np.abs(trace_cov_qoi) <= (self.rtol * b_norm) ** 2
        )
