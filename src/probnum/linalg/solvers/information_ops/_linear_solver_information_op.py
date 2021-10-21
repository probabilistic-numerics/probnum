"""Base class for linear solver information operators."""
import abc

import numpy as np

import probnum  # pylint: disable="unused-import"


class LinearSolverInformationOp(abc.ABC):
    r"""Information operator of a (probabilistic) linear solver.

    For a given action, the information operator collects information about the linear system to be solved.

    See Also
    --------
    MatVecInformationOp: Collect information via matrix-vector multiplication.
    ProjectedRHSInformationOp: Collect information via a projection of the current residual.
    """

    @abc.abstractmethod
    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> np.ndarray:
        """Return information about the linear system for a given solver state.

        Parameters
        ----------
        solver_state :
            Current state of the linear solver.
        """
        raise NotImplementedError
