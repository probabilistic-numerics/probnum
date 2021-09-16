"""Base class for linear solver information operators."""
import abc

import numpy as np

import probnum  # pylint: disable="unused-import"


class LinearSolverInfoOp(abc.ABC):
    r"""Information operator of a (probabilistic) linear solver.

    For a given action, the information operator collects information about the linear system to be solved.

    See Also
    --------
    MatVecInfoOp: Collect information via matrix-vector multiplication.
    ProjResidualInfoOp: Collect information via a projection of the current residual.
    """

    @abc.abstractmethod
    def __call__(
        self, solver_state: "probnum.linalg.solvers.ProbabilisticLinearSolverState"
    ) -> np.ndarray:
        """Return information about the linear system for a given solver state.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        raise NotImplementedError
